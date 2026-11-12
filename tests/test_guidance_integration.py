"""Hybrid rail/guidance/controller integration and orbital insertion acceptance tests."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.guidance import AutopilotConfig, GuidanceProgram, OrbitTarget, SteeringConfig
from src.physics import PRESETS, SimulationConfig, WindProfile, initial_state, simulate
from src.physics.constants import EARTH_OMEGA_RAD_S
from src.physics.frames import enu_basis, geodetic_to_ecef, rotation_z
from tests.guidance_cases import upper_vehicle, wind_tracking_case


@pytest.mark.parametrize("frame", ["ECI", "ENU"])
@pytest.mark.parametrize("dof", [3, 6])
def test_rail_constraint_and_exact_release(frame: str, dof: int) -> None:
    """Inclined rail constrains cross-track position to 1e-5 m until its terminal 15 m root."""
    vehicle = next(iter(PRESETS.values()))
    steering = SteeringConfig(rail_azimuth_rad=.7, rail_elevation_rad=1.4, vertical_hold_s=.5)
    config = SimulationConfig(duration_s=4., frame=frame, dof=dof, latitude_rad=.5, longitude_rad=-.8,
                              wind=WindProfile((0.,), ((25., -10., 3.),)))
    result = simulate(vehicle, config, guidance=GuidanceProgram(steering))
    release = [event for event in result.events if event.name == "rail_release"]
    assert len(release) == 1
    basis = enu_basis(config.latitude_rad, config.longitude_rad)
    origin_m = geodetic_to_ecef(config.latitude_rad, config.longitude_rad, config.launch_altitude_m)
    axis_ecef = basis @ steering.rail_axis_enu
    event = release[0]
    position_m = (rotation_z(-EARTH_OMEGA_RAD_S*event.time_s) @ event.state_before[:3] if frame == "ECI"
                  else origin_m+basis @ event.state_before[:3])
    assert_allclose(np.dot(position_m-origin_m, axis_ecef), 15., atol=1e-6)
    assert_allclose(event.state_before, event.state_after, atol=0.)
    for time_s, state in zip(result.time_s, result.state):
        if time_s > event.time_s:
            break
        offset_m = (rotation_z(-EARTH_OMEGA_RAD_S*time_s) @ state[:3]-origin_m if frame == "ECI" else basis @ state[:3])
        assert np.linalg.norm(offset_m-axis_ecef*np.dot(offset_m, axis_ecef)) < 1e-5
    assert np.all(np.diff(result.time_s) > 0.)


def test_ideal_gravity_turn_and_control_clock_independence() -> None:
    """Ideal alpha must vanish with wind, and accepted control clocks must not depend on solver steps."""
    vehicle = next(iter(PRESETS.values()))
    program = GuidanceProgram(SteeringConfig(vertical_hold_s=.2, pitch_duration_s=1., guidance_period_s=.5))
    config = SimulationConfig(duration_s=5., wind=WindProfile((0., 1000.), ((10., 2., 1.), (20., 5., 0.))))
    result = simulate(vehicle, config, guidance=program)
    comparison = simulate(vehicle, replace(config, max_step_s=.1), guidance=program)
    release_s = next(event.time_s for event in result.events if event.name == "rail_release")
    turning = result.time_s > release_s+1.2+1e-8
    assert np.max(np.abs(result.telemetry["alpha_rad"][turning])) < 1e-7
    regular_times_s = [row["time_s"] for row in result.guidance_records if row["event"] == "flight_update"]
    comparison_times_s = [row["time_s"] for row in comparison.guidance_records if row["event"] == "flight_update"]
    assert_allclose(regular_times_s, comparison_times_s, atol=2e-8, rtol=0.)
    assert_allclose(result.state[-1, :6], comparison.state[-1, :6], atol=1e-4, rtol=0.)


@pytest.mark.parametrize("controller_mode", ["PID", "LQR"])
@pytest.mark.parametrize("altitude_m", [15000., 80000.])
def test_wind_shear_tracking_and_physical_actuator_limits(controller_mode: str, altitude_m: float) -> None:
    """Track pitch/yaw through 3D shear: settled error <1 degree, travel <=5 deg, rate <=20 deg/s."""
    vehicle, config, program = wind_tracking_case(controller_mode, altitude_m)
    result = simulate(vehicle, config, guidance=program)
    assert result.state.shape[1] == 17
    assert np.max(result.telemetry["attitude_error_rad"][result.time_s > 20.]) < np.deg2rad(1.)
    assert np.max(np.abs(result.state[:, 15:17])) <= np.deg2rad(5.)+1e-7
    rates_rad_s = np.diff(result.state[:, 15:17], axis=0)/np.diff(result.time_s)[:, None]
    assert np.max(np.abs(rates_rad_s)) <= np.deg2rad(20.)+1e-5
    assert_allclose(result.telemetry["quaternion_norm"], 1., atol=2e-8)
    assert np.max(np.abs(result.state[:, 16])) > 1e-4  # yaw disturbance actually exercises the controller


@pytest.mark.parametrize("target", [OrbitTarget(), OrbitTarget(200000., 500000.)])
@pytest.mark.parametrize("dof", [3, 6])
def test_closed_loop_upper_stage_insertion(target: OrbitTarget, dof: int) -> None:
    """Insert into circular/elliptical targets with <100 m radius and <1 m/s speed errors."""
    config = SimulationConfig(duration_s=100., dof=dof, launch_altitude_m=196000., initial_elevation_rad=0.,
                              initial_velocity_enu_m_s=(7400., 0., 200.), atmosphere_enabled=False,
                              rotating_earth=False, j2_enabled=False)
    program = GuidanceProgram(SteeringConfig(rail_length_m=0., target=target, orbital_stage_index=0))
    result = simulate(upper_vehicle(), config, guidance=program)
    cutoff = [event for event in result.events if event.trigger == "guided_cutoff"]
    assert len(cutoff) == 1
    insertion = [record for record in result.guidance_records if record["phase"] == "insertion"]
    assert len(insertion) == 1 and insertion[0]["achieved"] is True
    errors = np.asarray(insertion[0]["errors_m_m_s_m_s"])
    assert abs(errors[0]) < 100. and np.max(np.abs(errors[1:])) < 1.
    assert cutoff[0].state_after[7] > 0.
    assert_allclose(cutoff[0].state_before, cutoff[0].state_after, atol=0.)
    assert np.all(result.telemetry["thrust_n"][result.time_s >= cutoff[0].time_s] == 0.)


@pytest.mark.parametrize("name,dof", [("Small-Lift Orbital Vehicle (Electron Class)", 3),
                                      ("Medium-Lift Two-Stage Orbital (Falcon 9 Class)", 3),
                                      ("Small-Lift Orbital Vehicle (Electron Class)", 6)])
def test_orbital_presets_from_launch(name: str, dof: int) -> None:
    """Orbital presets must reach their target through rail, staging, TVC, and guided SECO."""
    result = simulate(PRESETS[name], SimulationConfig(dof=dof, duration_s=460., max_step_s=5.),
                      guidance=GuidanceProgram(SteeringConfig(target=OrbitTarget())))
    events = [event.name for event in result.events]
    assert all(name in events for name in ("rail_release", "MECO", "stage_1_separation", "SES-1", "fairing_jettison", "SECO"))
    insertion = [record for record in result.guidance_records if record["phase"] == "insertion"]
    assert len(insertion) == 1 and insertion[0]["achieved"] is True


def test_rail_initialization_and_missing_tvc_arm_rejected() -> None:
    """Reject incompatible rail state and zero physical nozzle authority rather than inventing torque."""
    vehicle = upper_vehicle()
    config = SimulationConfig(duration_s=1., dof=6)
    invalid = replace(vehicle, stages=(replace(vehicle.stages[0], thrust_offset_body_m=(0., 0., 0.)),))
    with pytest.raises(ValueError, match="aft axial"):
        simulate(invalid, config, guidance=GuidanceProgram())
    state = initial_state(vehicle, replace(config, initial_velocity_enu_m_s=(1., 0., 0.)))
    with pytest.raises(ValueError, match="along-rail"):
        simulate(vehicle, config, state0=state, guidance=GuidanceProgram())


def test_guidance_reuse_determinism() -> None:
    """Reusing the same public program must produce identical run-local trajectories."""
    program = GuidanceProgram()
    config = SimulationConfig(duration_s=3.)
    vehicle = next(iter(PRESETS.values()))
    first = simulate(vehicle, config, guidance=program)
    second = simulate(vehicle, config, guidance=program)
    assert_allclose(first.time_s, second.time_s, atol=0., rtol=0.)
    assert_allclose(first.state, second.state, atol=0., rtol=0.)


def test_unreachable_mission_records_failed_insertion() -> None:
    """Exhaust an inadequate stage without a spurious guided cutoff or successful insertion."""
    vehicle = upper_vehicle()
    vehicle = replace(vehicle, stages=(replace(vehicle.stages[0], propellant_mass_kg=6., cutoff_time_s=2.),))
    config = SimulationConfig(duration_s=3., launch_altitude_m=196000.,
                              initial_velocity_enu_m_s=(7400., 0., 200.), atmosphere_enabled=False,
                              rotating_earth=False, j2_enabled=False)
    program = GuidanceProgram(SteeringConfig(rail_length_m=0., target=OrbitTarget(2000000.), orbital_stage_index=0))
    result = simulate(vehicle, config, guidance=program)
    assert not any(event.trigger == "guided_cutoff" for event in result.events)
    insertion = [row for row in result.guidance_records if row["phase"] == "insertion"]
    assert len(insertion) == 1 and insertion[0]["achieved"] is False
    assert insertion[0]["cutoff_trigger"] in ("scheduled_cutoff", "propellant_depletion")
    assert not any(row["feasible"] for row in result.guidance_records if row["phase"] == "orbital_plan")


def test_guided_six_dof_frame_consistency_and_tolerance_convergence() -> None:
    """Rotating ECI/ENU guidance and loose/tight solves agree within 2 mm and 0.2 mm/s."""
    vehicle = next(iter(PRESETS.values()))
    config = SimulationConfig(duration_s=8., dof=6, latitude_rad=.5, longitude_rad=-.8, rtol=1e-9,
                              wind=WindProfile((0., 1000.), ((10., -5., 1.), (20., 5., -1.))))
    program = GuidanceProgram(SteeringConfig(vertical_hold_s=.5, pitch_duration_s=4.))
    reference = simulate(vehicle, config, guidance=program)
    for variant in (replace(config, frame="ENU"), replace(config, rtol=1e-6)):
        result = simulate(vehicle, variant, guidance=program)
        assert_allclose(result.telemetry["position_eci_m"][-1], reference.telemetry["position_eci_m"][-1], atol=.002, rtol=0.)
        assert_allclose(result.telemetry["velocity_eci_m_s"][-1], reference.telemetry["velocity_eci_m_s"][-1], atol=.0002, rtol=0.)
        assert_allclose(result.state[-1, 12:17], reference.state[-1, 12:17], atol=1e-6, rtol=0.)
        assert_allclose(result.events[0].time_s, reference.events[0].time_s, atol=1e-6, rtol=0.)
