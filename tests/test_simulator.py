"""Event, frame, conservation, and rigid-body verification with declared tolerances."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.physics.constants import EARTH_MU_M3_S2, EARTH_OMEGA_RAD_S, G0_M_S2, WGS84_A_M
from src.physics.frames import quaternion_matrix, rotation_z
from src.physics.simulator import (ControlInput, SimulationConfig, initial_state, orbital_elements, simulate)
from src.physics.vehicle import StageConfig, VehicleConfig


def _vehicle() -> VehicleConfig:
    """Return an isolated one-stage 100 kg propellant, 20 kg final-mass test vehicle."""
    return VehicleConfig("analytical", (StageConfig("motor", 10., 100., 300., 0., mass_flow_kg_s=10.),), 10.)


def _vacuum_config(**overrides: object) -> SimulationConfig:
    """Return a 100 m initial-altitude force-isolation configuration in SI units."""
    return replace(SimulationConfig(duration_s=12., launch_altitude_m=100., atmosphere_enabled=False,
                                    mu_m3_s2=0., j2_enabled=False, rotating_earth=False), **overrides)


def _x_axis(time_s: float, state: np.ndarray) -> np.ndarray:
    """Return constant +x unit axis (dimensionless); time [s] and SI state are unused."""
    return np.array([1., 0., 0.])


@pytest.mark.parametrize("method", ["DOP853", "RK45"])
def test_tsiolkovsky_and_exact_cutoff(method: str) -> None:
    """Vacuum delta-V must match Tsiolkovsky within 2e-5 m/s; cutoff time within 1e-8 s."""
    vehicle = _vehicle()
    result = simulate(vehicle, _vacuum_config(method=method), thrust_direction=_x_axis)
    expected_delta_v_m_s = 300.*G0_M_S2*np.log(120./20.)
    assert_allclose(result.state[-1, 3:6], [expected_delta_v_m_s, 0., 0.], atol=2e-5, rtol=0.)
    assert_allclose(result.state[-1, 6], 20., atol=1e-6)
    assert_allclose(result.state[-1, 7], 0., atol=1e-6)
    assert_allclose(result.events[0].time_s, 10., atol=1e-8)
    assert np.all(result.telemetry["thrust_n"][result.time_s >= result.events[0].time_s] == 0.)
    assert np.all(np.diff(result.time_s) > 0.)


def test_staging_coast_residual_fuel_and_coincident_fairing() -> None:
    """Verify ordered roots and exact mass jumps, including residual fuel and a coincident fairing."""
    booster = StageConfig("booster", 10., 100., 250., 0., mass_flow_kg_s=10., cutoff_time_s=5.,
                          separation_delay_s=2., coast_after_separation_s=3.)
    upper = StageConfig("upper", 5., 20., 300., 0., mass_flow_kg_s=10.)
    vehicle = VehicleConfig("two-stage", (booster, upper), 2., fairing_mass_kg=3., fairing_jettison_time_s=5.)
    result = simulate(vehicle, _vacuum_config(duration_s=14.), thrust_direction=_x_axis)
    assert [event.name for event in result.events] == ["MECO", "fairing_jettison", "stage_1_separation", "SES-1", "SECO"]
    assert_allclose([event.time_s for event in result.events], [5., 5., 7., 10., 12.], atol=1e-8)
    for event in result.events:
        assert_allclose(event.state_before[:6], event.state_after[:6], atol=0.)
    separation = result.events[2]
    assert_allclose(separation.state_before[6] - separation.state_after[6], 60., atol=1e-7)
    assert_allclose(separation.state_after[7], 20.)
    assert_allclose(result.state[-1, 6], 7., atol=1e-7)
    coast = (result.time_s > 5.) & (result.time_s < 10.)
    assert not np.any(result.engine_on[coast])
    assert np.all(result.telemetry["thrust_n"][coast] == 0.)
    assert np.all(np.diff(result.state[:, 6]) <= 1e-7)


def test_zero_delay_and_zero_time_fairing() -> None:
    """Zero-duration transition chains must resolve once each with right-continuous rows."""
    stage = StageConfig("motor", 10., 20., 300., 0., mass_flow_kg_s=10.)
    vehicle = VehicleConfig("instant", (stage, stage), 2., 3., fairing_jettison_time_s=0.)
    result = simulate(vehicle, _vacuum_config(duration_s=5.))
    assert [event.name for event in result.events] == ["fairing_jettison", "MECO", "stage_1_separation", "SES-1", "SECO"]
    assert_allclose([event.time_s for event in result.events], [0., 2., 2., 2., 4.], atol=1e-8)
    assert_allclose(result.state[0, 6], vehicle.initial_mass_kg - 3.)
    assert np.all(np.diff(result.time_s) > 0.)


def test_curve_fuel_depletion_before_scheduled_cutoff() -> None:
    """Linear ramp consumes 10 kg at sqrt(2) s, before its scheduled 4 s cutoff."""
    stage = StageConfig("ramp", 100., 10., 200., 0., thrust_curve_time_s=(0., 4.),
                        thrust_curve_vac_n=(0., 40.*200.*G0_M_S2))
    result = simulate(VehicleConfig("ramp", (stage,), 0.), _vacuum_config(duration_s=3., max_step_s=.1))
    assert result.events[0].trigger == "propellant_depletion"
    assert_allclose(result.events[0].time_s, np.sqrt(2.), atol=1e-8)
    assert_allclose(result.state[-1, 3], 200.*G0_M_S2*np.log(110./100.), atol=1e-6)
    assert_allclose(result.state[-1, 6], 100., atol=1e-7)


def test_curve_end_cutoff_with_residual_propellant() -> None:
    """Thrust-curve end is a terminal cutoff, not an interpolation-driven engine switch."""
    stage = StageConfig("curve", 100., 100., 200., 0., thrust_curve_time_s=(0., 2., 4.),
                        thrust_curve_vac_n=(0., 4000., 0.))
    result = simulate(VehicleConfig("curve", (stage,), 0.), _vacuum_config(duration_s=6., max_step_s=.1))
    assert result.events[0].trigger == "scheduled_cutoff"
    assert_allclose(result.events[0].time_s, 4., atol=1e-8)
    assert_allclose(result.state[-1, 7], 100. - 8000./(200.*G0_M_S2), atol=1e-5)


def test_altitude_fairing_and_ground_impact() -> None:
    """Known constant-speed height crossing and descending impact roots must occur to 1e-6 s."""
    vehicle = replace(_vehicle(), fairing_mass_kg=3., fairing_jettison_altitude_m=150.)
    config = _vacuum_config(duration_s=20., engines_enabled=False, initial_velocity_enu_m_s=(0., 0., 10.))
    result = simulate(vehicle, config)
    assert result.events[0].name == "fairing_jettison"
    assert_allclose(result.events[0].time_s, 5., atol=1e-6)
    descent = simulate(_vehicle(), replace(config, initial_velocity_enu_m_s=(0., 0., -10.)))
    assert descent.termination == "ground_impact"
    assert_allclose(descent.time_s[-1], 10., atol=1e-6)
    assert_allclose(descent.telemetry["altitude_m"][-1], 0., atol=1e-6)
    assert descent.events[-1].name == "ground_impact"


@pytest.mark.parametrize("frame", ["ECI", "ENU"])
def test_launch_from_surface_does_not_trigger_impact(frame: str) -> None:
    """A supported launch ascends; an engine-off vehicle at rest hits the surface at t=0."""
    config = SimulationConfig(duration_s=1., frame=frame, rotating_earth=False)
    ascent = simulate(_vehicle(), config)
    assert ascent.termination == "duration"
    assert ascent.telemetry["altitude_m"][-1] > 1.
    unsupported = simulate(_vehicle(), replace(config, engines_enabled=False))
    assert unsupported.termination == "ground_impact"
    assert unsupported.time_s[-1] == 0.


@pytest.mark.parametrize("rotating", [True, False])
def test_eci_enu_equivalence_with_j2_wind_and_thrust(rotating: bool) -> None:
    """Alternative frames must agree within 2 mm and 2e-5 m/s for a powered flight."""
    from src.physics.aerodynamics import WindProfile

    config = SimulationConfig(duration_s=15., launch_altitude_m=100., latitude_rad=.6, longitude_rad=-1.,
                              rotating_earth=rotating, initial_velocity_enu_m_s=(20., 10., 0.),
                              wind=WindProfile((0., 10000.), ((10., -5., 1.), (30., 10., 0.))))
    eci = simulate(_vehicle(), config)
    enu = simulate(_vehicle(), replace(config, frame="ENU"))
    assert_allclose(eci.telemetry["position_eci_m"][-1], enu.telemetry["position_eci_m"][-1], atol=.002, rtol=0.)
    assert_allclose(eci.telemetry["velocity_eci_m_s"][-1], enu.telemetry["velocity_eci_m_s"][-1], atol=2e-5, rtol=0.)
    assert_allclose(eci.telemetry["q_pa"][-1], enu.telemetry["q_pa"][-1], rtol=1e-7)


def test_orbital_elements_degenerate_and_unbound_cases() -> None:
    """Circular, inclined, hyperbolic, and radial orbits must have deliberate diagnostics."""
    radius_m = WGS84_A_M + 400000.
    speed_m_s = np.sqrt(EARTH_MU_M3_S2 / radius_m)
    result = orbital_elements([radius_m, 0., 0.], [0., speed_m_s/np.sqrt(2.), speed_m_s/np.sqrt(2.)])
    assert_allclose(result["semi_major_axis_m"], radius_m, atol=1e-6)
    assert_allclose(result["eccentricity"], 0., atol=1e-14)
    assert_allclose(result["inclination_rad"], np.pi/4., atol=1e-14)
    assert_allclose(result["perigee_altitude_m"], 400000., atol=1e-6)
    assert_allclose(result["apogee_altitude_m"], 400000., atol=1e-6)
    hyperbolic = orbital_elements([radius_m, 0., 0.], [0., 2.*speed_m_s, 0.])
    assert hyperbolic["eccentricity"] > 1. and np.isinf(hyperbolic["apogee_altitude_m"])
    radial = orbital_elements([radius_m, 0., 0.], [10., 0., 0.])
    assert np.isnan(radial["inclination_rad"])


def test_one_orbit_energy_conservation_and_frame_equivalence() -> None:
    """One circular orbit conserves energy to 2e-9 relative and closes position within 0.2 m."""
    radius_m = WGS84_A_M + 400000.
    speed_m_s = np.sqrt(EARTH_MU_M3_S2/radius_m)
    period_s = 2.*np.pi*radius_m/speed_m_s
    config = SimulationConfig(duration_s=period_s, launch_altitude_m=400000., atmosphere_enabled=False,
                              engines_enabled=False, j2_enabled=False, max_step_s=60.,
                              initial_velocity_enu_m_s=(speed_m_s - EARTH_OMEGA_RAD_S*radius_m, 0., 0.))
    result = simulate(_vehicle(), config)
    energy_m2_s2 = result.telemetry["specific_energy_m2_s2"]
    assert np.max(np.abs(energy_m2_s2/energy_m2_s2[0] - 1.)) < 2e-9
    assert_allclose(result.state[-1, :3], result.state[0, :3], atol=.2, rtol=0.)
    rotating = simulate(_vehicle(), replace(config, frame="ENU"))
    assert_allclose(rotating.telemetry["position_eci_m"][-1], result.state[-1, :3], atol=.2, rtol=0.)


def test_6dof_principal_axis_rotation_and_constant_torque() -> None:
    """Quaternion spin follows the analytic rotation and a sphere obeys torque/I [rad/s^2]."""
    vehicle = _vehicle()
    config = _vacuum_config(duration_s=2., engines_enabled=False, dof=6, initial_body_rate_rad_s=(.5, 0., 0.))
    result = simulate(vehicle, config)
    expected_body_rotation = _rotation_from_x(1.)
    assert_allclose(quaternion_matrix(result.state[-1, 8:12]), quaternion_matrix(result.state[0, 8:12]) @ expected_body_rotation, atol=1e-9)
    assert_allclose(result.telemetry["quaternion_norm"], 1., atol=1e-9)
    sphere = replace(vehicle, stages=(replace(vehicle.stages[0], inertia_per_mass_m2=(1., 1., 1.)),))
    torque_result = simulate(sphere, replace(config, initial_body_rate_rad_s=(0., 0., 0.)), control=_constant_torque)
    assert_allclose(torque_result.state[-1, 12:15], [0., 0., 2./120.], atol=1e-10)


def _rotation_from_x(angle_rad: float) -> np.ndarray:
    """Return independent analytic x-axis rotation matrix (dimensionless)."""
    return np.array([[1., 0., 0.], [0., np.cos(angle_rad), -np.sin(angle_rad)], [0., np.sin(angle_rad), np.cos(angle_rad)]])


def _constant_torque(time_s: float, state: np.ndarray) -> ControlInput:
    """Apply 1 N*m about body +z; time [s] and SI state are unused."""
    return ControlInput(torque_body_n_m=(0., 0., 1.))


def test_6dof_torque_free_energy_and_angular_momentum() -> None:
    """Asymmetric rigid-body coast conserves kinetic energy and inertial momentum to 1e-8."""
    vehicle = _vehicle()
    vehicle = replace(vehicle, stages=(replace(vehicle.stages[0], inertia_per_mass_m2=(1., 1.5, 2.)),))
    result = simulate(vehicle, _vacuum_config(duration_s=10., engines_enabled=False, dof=6,
                                             initial_body_rate_rad_s=(.3, .4, .5), max_step_s=.2))
    inertia_kg_m2 = 120.*np.array([1., 1.5, 2.])
    energy_j = .5*np.sum(inertia_kg_m2*result.state[:, 12:15]**2, axis=1)
    momentum_kg_m2_s = np.array([quaternion_matrix(state[8:12]) @ (inertia_kg_m2*state[12:15]) for state in result.state])
    assert_allclose(energy_j, energy_j[0], rtol=1e-8)
    assert_allclose(momentum_kg_m2_s, np.broadcast_to(momentum_kg_m2_s[0], momentum_kg_m2_s.shape), rtol=1e-8, atol=1e-8)


def test_6dof_rotating_frame_kinematics_and_translations() -> None:
    """Body-to-ECI attitude and translation agree between ECI and rotating ENU."""
    config = SimulationConfig(duration_s=3., dof=6, launch_altitude_m=100., latitude_rad=.5,
                              initial_elevation_rad=1.4, initial_body_rate_rad_s=(.1, 0., 0.))
    eci = simulate(_vehicle(), config)
    enu = simulate(_vehicle(), replace(config, frame="ENU"))
    from src.physics.frames import enu_basis

    enu_to_eci = rotation_z(EARTH_OMEGA_RAD_S * config.duration_s) @ enu_basis(config.latitude_rad, config.longitude_rad)
    assert_allclose(quaternion_matrix(eci.state[-1, 8:12]), enu_to_eci @ quaternion_matrix(enu.state[-1, 8:12]), atol=1e-8)
    assert_allclose(eci.state[-1, :3], enu.telemetry["position_eci_m"][-1], atol=.001, rtol=0.)


def test_tolerance_convergence() -> None:
    """Tighter RK45 tolerances must improve Tsiolkovsky error and stay within 2e-5 m/s."""
    vehicle = _vehicle()
    exact_m_s = 300.*G0_M_S2*np.log(6.)
    loose = simulate(vehicle, _vacuum_config(method="RK45", rtol=1e-4, max_step_s=3.), thrust_direction=_x_axis)
    tight = simulate(vehicle, _vacuum_config(method="RK45", rtol=1e-10, max_step_s=3.), thrust_direction=_x_axis)
    assert abs(tight.state[-1, 3]-exact_m_s) < abs(loose.state[-1, 3]-exact_m_s)
    assert abs(tight.state[-1, 3]-exact_m_s) < 2e-5


def test_invalid_state_and_control() -> None:
    """Reject invalid state dimensions, inconsistent initial mass, and mixed DoF callbacks."""
    config = _vacuum_config()
    with pytest.raises(ValueError):
        simulate(_vehicle(), config, state0=np.zeros(7))
    state = initial_state(_vehicle(), config)
    state[6] = 1.
    with pytest.raises(ValueError):
        simulate(_vehicle(), config, state0=state)
    with pytest.raises(ValueError):
        simulate(_vehicle(), config, control=_constant_torque)
    with pytest.raises(ValueError):
        ControlInput(thrust_direction_body=(2., 0., 0.))


def test_6dof_variable_inertia_balance_and_staging_continuity() -> None:
    """Verify the declared zero-efflux-angular-momentum model and impulsive mass-only staging."""
    stage = StageConfig("test", 10., 20., 300., 0., mass_flow_kg_s=10.)
    vehicle = VehicleConfig("rotating stage test", (stage, stage), 10., 2., fairing_jettison_time_s=1.)
    result = simulate(vehicle, _vacuum_config(duration_s=5., dof=6, initial_body_rate_rad_s=(.1, 0., 0.)))
    for event in result.events:
        assert_allclose(event.state_before[8:], event.state_after[8:], atol=0.)
    first_segment = result.time_s < 1.
    axial_momentum_kg_m2_s = result.state[first_segment, 6]*stage.principal_inertia_per_mass_m2[0]*result.state[first_segment, 12]
    assert_allclose(axial_momentum_kg_m2_s, axial_momentum_kg_m2_s[0], rtol=1e-8)
    assert_allclose(result.telemetry["quaternion_norm"], 1., atol=1e-8)


def test_6dof_tvc_force_and_moment_are_coupled() -> None:
    """Off-axis nozzle force produces the expected initial angular acceleration [rad/s^2]."""
    stage = StageConfig("offset", 10., 100., 300., 0., mass_flow_kg_s=10.,
                        inertia_per_mass_m2=(1., 1., 1.), thrust_offset_body_m=(-1., 0., 0.))
    vehicle = VehicleConfig("TVC", (stage,), 10.)
    config = _vacuum_config(duration_s=1e-4, dof=6)
    result = simulate(vehicle, config, control=_gimbaled_thrust)
    expected_rate_rad_s = -10.*300.*G0_M_S2*np.sin(.01)/120.*config.duration_s
    assert_allclose(result.state[-1, 14], expected_rate_rad_s, rtol=2e-5)
    assert result.state[-1, 14] < 0.


def _gimbaled_thrust(time_s: float, state: np.ndarray) -> ControlInput:
    """Command a 0.01 rad body x-y thrust deflection without an extra torque."""
    return ControlInput(thrust_direction_body=(float(np.cos(.01)), float(np.sin(.01)), 0.))


def test_ground_impact_wins_over_coincident_fairing() -> None:
    """A terminating impact must not restart to process a simultaneously scheduled jettison."""
    vehicle = replace(_vehicle(), fairing_mass_kg=2., fairing_jettison_time_s=10.)
    result = simulate(vehicle, _vacuum_config(duration_s=15., engines_enabled=False, initial_velocity_enu_m_s=(0., 0., -10.)))
    assert result.termination == "ground_impact"
    assert result.events[-1].name == "ground_impact"
    assert not any(event.name == "fairing_jettison" for event in result.events)


def test_three_stage_event_chain() -> None:
    """The scheduler supports arbitrary stage counts without hard-coded two-stage mass logic."""
    stage = StageConfig("repeat", 10., 10., 300., 0., mass_flow_kg_s=10., coast_after_separation_s=.5)
    result = simulate(VehicleConfig("three stages", (stage, stage, stage), 5.), _vacuum_config(duration_s=5.))
    assert [event.name for event in result.events] == ["MECO", "stage_1_separation", "SES-1", "SECO",
                                                       "stage_2_separation", "stage_3_ignition", "stage_3_cutoff"]
    assert_allclose([event.time_s for event in result.events], [1., 1., 1.5, 2.5, 2.5, 3., 4.], atol=1e-8)
    assert_allclose(result.state[-1, 6], 15., atol=1e-7)
