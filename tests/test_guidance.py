"""Unit tests for phased steering, vacuum shooting, and feasibility reporting."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from src.guidance import AutopilotConfig, TVCAutopilot, attitude_kinematics
from src.physics import SimulationConfig, simulate
from tests.guidance_cases import wind_tracking_case

from src.guidance import GuidanceProgram, OrbitTarget, SteeringConfig, gravity_turn_direction, pitch_kick, solve_linear_tangent
from src.guidance.steering import available_burn_s
from src.physics.constants import EARTH_MU_M3_S2, G0_M_S2, WGS84_A_M
from src.physics.vehicle import StageConfig
from tests.guidance_cases import navigation_sample, upper_vehicle


@pytest.mark.parametrize("profile", ["linear", "exponential"])
def test_pitch_profile_boundaries_and_monotonicity(profile: str) -> None:
    """Pitch [rad] must be bounded, continuous, and monotonic across its timed maneuver."""
    values_rad = np.array([pitch_kick(time_s, 5., 10., 1.5, .1, profile) for time_s in np.linspace(0., 20., 1001)])
    assert_allclose(values_rad[0], 1.5)
    assert_allclose(values_rad[-1], 1.4)
    assert np.max(np.diff(values_rad)) <= 1e-14
    assert np.max(np.abs(np.diff(values_rad))) < .001
    if profile == "exponential":
        assert abs(pitch_kick(5.00001, 5., 10., 1.5, .1)-1.5) < 1e-10
        assert abs(pitch_kick(14.99999, 5., 10., 1.5, .1)-1.4) < 1e-10
    else:
        assert_allclose(pitch_kick(10., 5., 10., 1.5, .1, profile), 1.45)


def test_zero_lift_uses_air_relative_velocity() -> None:
    """Nose alignment must cancel all three wind components, including vertical wind [m/s]."""
    velocity_m_s, wind_m_s = np.array([20., 30., 100.]), np.array([10., -5., 8.])
    relative_m_s = velocity_m_s-wind_m_s
    command = gravity_turn_direction(relative_m_s, np.array([0., 0., 1.]))
    assert_allclose(np.cross(command, relative_m_s), 0., atol=1e-12)
    assert_allclose(np.linalg.norm(command), 1.)
    assert_allclose(gravity_turn_direction(np.zeros(3), command), command)


def test_remaining_burn_curve_and_constant_flow() -> None:
    """Fuel-bounded burn time [s] must respect exact piecewise-linear flow integrals."""
    stage = upper_vehicle().stages[0]
    assert_allclose(available_burn_s(stage, 10., 30.), 10.)
    assert_allclose(available_burn_s(stage, 99., 300.), 1.)
    ramp = StageConfig("ramp", 100., 10., 200., 0., thrust_curve_time_s=(0., 4.),
                       thrust_curve_vac_n=(0., 40.*200.*G0_M_S2))
    assert_allclose(available_burn_s(ramp, 0., 10.), np.sqrt(2.), atol=1e-10)
    assert available_burn_s(stage, 0., 0.) == 0.


def test_orbit_targets_vis_viva() -> None:
    """Circular and elliptical insertion speeds [m/s] must obey independent energy formulas."""
    radius_m = WGS84_A_M+200000.
    circular = OrbitTarget()
    elliptical = OrbitTarget(200000., 500000.)
    assert_allclose(circular.speed_m_s(EARTH_MU_M3_S2), np.sqrt(EARTH_MU_M3_S2/radius_m))
    assert_allclose(elliptical.speed_m_s(EARTH_MU_M3_S2), np.sqrt(EARTH_MU_M3_S2*(2./radius_m-1./(radius_m+150000.))))
    with pytest.raises(ValueError):
        OrbitTarget(200000., 100000.)


def test_linear_tangent_solution_and_analytic_pitch_rate() -> None:
    """The numerical shooter must find a feasible bounded burn and consistent pitch rate [rad/s]."""
    navigation = navigation_sample()
    solution = solve_linear_tangent(navigation, upper_vehicle().stages[0], OrbitTarget())
    assert solution.feasible
    assert np.max(np.abs(solution.errors)) < .1
    assert 0. < solution.duration_s < 100.
    time_s = .4*solution.duration_s
    numeric_rad_s = (solution.pitch_rad(time_s+.001)-solution.pitch_rad(time_s-.001))/.002
    assert_allclose(numeric_rad_s, solution.pitch_rate_rad_s(time_s), rtol=1e-7)
    assert_allclose(np.linalg.norm(solution.direction(navigation)), 1., atol=1e-12)


def test_unreachable_target_reported_without_cutoff() -> None:
    """An insufficient-fuel target must be reported as infeasible rather than a solved orbit."""
    navigation = replace(navigation_sample(), propellant_kg=1.)
    program = GuidanceProgram(SteeringConfig(rail_length_m=0., target=OrbitTarget(2000000.), orbital_stage_index=0))
    session = program.create_session(upper_vehicle(), 3)
    command = session.update(navigation)
    assert command.phase == "orbital_infeasible"
    assert np.isinf(command.cutoff_time_s)
    assert session.records[0]["feasible"] is False


def test_program_memory_is_per_run_and_command_is_pure() -> None:
    """Factory sessions and accepted updates must not leak mutable memory into RHS evaluations."""
    program = GuidanceProgram(SteeringConfig(rail_length_m=0.))
    first = program.create_session(upper_vehicle(), 3)
    second = program.create_session(upper_vehicle(), 3)
    nav = navigation_sample()
    command = first.update(nav)
    first_axis = command.direction_eci(nav)
    command.direction_eci(replace(nav, time_s=3.))
    assert_allclose(command.direction_eci(nav), first_axis, atol=0.)
    assert len(first.records) == 1 and len(second.records) == 0


@pytest.mark.parametrize("changes", [{"rail_length_m": -1.}, {"guidance_period_s": 0.},
                                     {"pitch_duration_s": 0.}, {"rail_elevation_rad": 0.}, {"pitch_profile": "bad"}])
def test_invalid_steering_configuration(changes: dict[str, object]) -> None:
    """Invalid physical settings must fail before propagation."""
    with pytest.raises(ValueError):
        SteeringConfig(**changes)


@pytest.mark.parametrize("method", ["DOP853", "RK45"])
def test_integrated_quaternion_unit_norm(method: str) -> None:
    """Preserve unit norm within 1e-9 over 120 s and match analytical constant-axis rotation."""
    rate_rad_s = np.array([.1, -.2, .3])

    def derivative(time_s: float, quaternion: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return quaternion derivative [1/s] for fixed body rate [rad/s]."""
        return attitude_kinematics(quaternion, rate_rad_s)

    times_s = np.linspace(0., 120., 601)
    result = solve_ivp(derivative, (0., 120.), [1., 0., 0., 0.], method=method,
                       t_eval=times_s, rtol=1e-11, atol=1e-13)
    assert result.success
    assert_allclose(np.linalg.norm(result.y, axis=0), 1., rtol=0., atol=1e-9)
    half_angle_rad = .5*np.linalg.norm(rate_rad_s)*times_s
    expected = np.vstack((np.cos(half_angle_rad),
                          rate_rad_s[:, None]/np.linalg.norm(rate_rad_s)*np.sin(half_angle_rad)))
    assert_allclose(result.y, expected, rtol=0., atol=1e-9)


@pytest.mark.parametrize("mode", ["PID", "LQR"])
@pytest.mark.parametrize("axis", [1, 2])
@pytest.mark.parametrize("sign", [-1., 1.])
def test_both_gimbal_axes_reach_signed_saturation(mode: str, axis: int, sign: float) -> None:
    """Force each pitch/yaw travel stop in both directions at exactly five degrees [rad]."""
    controller = TVCAutopilot(AutopilotConfig(mode=mode))
    direction = np.zeros(3)
    direction[axis] = sign
    command_rad = controller.command(0., np.eye(3), np.zeros(3), direction,
                                     np.full(3, 1000.), np.zeros(3), 1., 1., np.zeros(2))
    active_axis = 1 if axis == 1 else 0
    assert controller.saturated
    # For an aft nozzle, positive body-z steering requires positive pitch;
    # positive body-y steering requires negative yaw (offset cross force).
    expected_sign = -sign if axis == 1 else sign
    assert_allclose(command_rad[active_axis], expected_sign*np.deg2rad(5.), rtol=0., atol=1e-12)
    assert_allclose(command_rad[1-active_axis], 0., atol=1e-12)


@pytest.mark.parametrize("mode", ["PID", "LQR"])
def test_attitude_step_settles_through_wind_gusts(mode: str) -> None:
    """A five-degree initial command step settles below one degree by 20 s during 3D shear."""
    vehicle, config, program = wind_tracking_case(mode)
    # Hold a fixed vertical command for the entire experiment: no moving
    # pitch/gravity-turn reference can conceal step-response settling.
    program = replace(program, steering=replace(program.steering, vertical_hold_s=30.))
    result = simulate(vehicle, config, guidance=program)
    error_rad = result.telemetry["attitude_error_rad"]
    assert_allclose(error_rad[0], np.deg2rad(5.), rtol=0., atol=1e-10)
    assert np.max(error_rad[result.time_s >= 20.]) < np.deg2rad(1.)
    assert np.max(np.abs(result.state[:, 15:17])) <= np.deg2rad(5.)+1e-7
    assert_allclose(np.linalg.norm(result.state[:, 8:12], axis=1), 1., rtol=0., atol=2e-8)
    # Ensure the vehicle traversed the shear and yaw control was exercised.
    assert result.telemetry["altitude_m"].max() > 18000.
    assert np.max(np.abs(result.state[:, 16])) > 1e-6


@pytest.mark.parametrize("apogee_m", [200000., 500000.])
def test_orbital_insertion_against_independent_state_metrics(apogee_m: float) -> None:
    """Verify actual ECI cutoff radius and vis-viva speed, independent of success flags."""
    target = OrbitTarget(200000., apogee_m)
    config = SimulationConfig(duration_s=100., dof=6, launch_altitude_m=196000.,
                              initial_elevation_rad=0., initial_velocity_enu_m_s=(7400., 0., 200.),
                              atmosphere_enabled=False, rotating_earth=False, j2_enabled=False)
    program = GuidanceProgram(SteeringConfig(rail_length_m=0., target=target, orbital_stage_index=0))
    result = simulate(upper_vehicle(), config, guidance=program)
    cutoff = [event for event in result.events if event.trigger == "guided_cutoff"]
    assert len(cutoff) == 1
    position_m, velocity_m_s = cutoff[0].state_before[:3], cutoff[0].state_before[3:6]
    radius_m = np.linalg.norm(position_m)
    radial_m_s = np.dot(position_m, velocity_m_s)/radius_m
    tangent_m_s = np.linalg.norm(velocity_m_s-radial_m_s*position_m/radius_m)
    target_radius_m = WGS84_A_M+200000.
    semi_major_m = .5*(target_radius_m+WGS84_A_M+apogee_m)
    target_speed_m_s = np.sqrt(EARTH_MU_M3_S2*(2./target_radius_m-1./semi_major_m))
    assert abs(radius_m-target_radius_m) < 100.
    assert abs(radial_m_s) < 1.
    assert abs(tangent_m_s-target_speed_m_s) < 1.
    assert cutoff[0].state_before[7] > 0.
