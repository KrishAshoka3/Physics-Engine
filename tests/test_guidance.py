"""Unit tests for phased steering, vacuum shooting, and feasibility reporting."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

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
