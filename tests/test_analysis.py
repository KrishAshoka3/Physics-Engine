"""Phase 3 acceptance: RNG replay, covariance geometry, and converged constrained sizing."""

from dataclasses import asdict, replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from src.analysis import (OptimizationConfig, covariance_ellipses, default_dispersion_space,
                          optimize_trajectory, run_monte_carlo, sized_vehicle, trajectory_metrics)
from src.physics import simulate
from tests.analysis_cases import short_launch


def test_fixed_generator_seed_parallel_monte_carlo_determinism() -> None:
    """Seeded Generator streams reproduce every input, outcome, and metric across worker counts."""
    vehicle, config, guidance = short_launch()
    root_seed = 271828
    first = run_monte_carlo(vehicle, config, guidance, count=6, seed=root_seed, workers=2)
    repeated = run_monte_carlo(vehicle, config, guidance, count=6, seed=root_seed, workers=2)
    serial = run_monte_carlo(vehicle, config, guidance, count=6, seed=root_seed, workers=1)
    space = default_dispersion_space(2)
    replay_rows = []
    for run in first.runs:
        generator: np.random.Generator = np.random.default_rng(run.seed)
        unit_row = generator.uniform(np.finfo(float).eps, 1.-np.finfo(float).eps, len(space.parameters))
        replay_rows.append(unit_row)
    assert_array_equal(first.inputs, space.transform(replay_rows))
    assert len({run.seed for run in first.runs}) == 6
    for comparison in (repeated, serial):
        assert_array_equal(first.inputs, comparison.inputs)
        for expected, actual in zip(first.runs, comparison.runs):
            assert expected.index == actual.index and expected.seed == actual.seed
            assert expected.status == actual.status == "impact"
            assert expected.error is actual.error is None
            assert_allclose(list(asdict(expected.metrics).values()), list(asdict(actual.metrics).values()),
                            rtol=0., atol=0., equal_nan=True)
        for expected, actual in zip(first.ellipses, comparison.ellipses):
            assert_array_equal(expected.covariance_m2, actual.covariance_m2)
            assert_array_equal(expected.points_m, actual.points_m)


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_covariance_psd_and_exact_sigma_axis_scaling(rank: int) -> None:
    """Covariance is PSD and semiaxes scale exactly 1:2:3, including degenerate clouds."""
    generator: np.random.Generator = np.random.default_rng(314159)
    points_m = generator.normal(size=(128, 2)) @ np.array([[30., 7.], [0., 4.]])
    if rank == 1:
        points_m[:, 1] = 0.
    elif rank == 0:
        points_m[:] = 0.
    points_m += [1000., -200.]
    ellipses = covariance_ellipses(points_m)
    covariance_m2 = ellipses[0].covariance_m2
    assert_allclose(covariance_m2, covariance_m2.T, atol=0., rtol=0.)
    eigenvalues_m2 = np.linalg.eigvalsh(covariance_m2)
    assert np.min(eigenvalues_m2) >= -1e-12*max(1., np.max(eigenvalues_m2))
    assert ellipses[0].rank == rank
    base_axes_m = np.sqrt(np.maximum(eigenvalues_m2[::-1], 0.))
    for sigma, ellipse in enumerate(ellipses, 1):
        assert_array_equal(ellipse.covariance_m2, covariance_m2)
        assert_allclose(ellipse.semiaxes_m, sigma*base_axes_m, rtol=1e-14, atol=1e-12)
        assert_allclose(ellipse.points_m-ellipse.center_m,
                        sigma*(ellipses[0].points_m-ellipses[0].center_m), rtol=0., atol=1e-12)


def test_optimizer_converges_and_independent_trajectory_improves() -> None:
    """SLSQP must converge, improve apogee by >1 m, and pass a separate finer load verification."""
    vehicle, config, guidance = short_launch()
    settings = OptimizationConfig(stage1_fraction_bounds=(.35, .8), pitch_kick_bounds_rad=(0., .1),
                                   max_q_pa=5000., max_axial_g=5., max_iterations=30, max_evaluations=140)
    optimized = optimize_trajectory(vehicle, config, guidance, settings)
    assert optimized.optimizer_converged, optimized.message
    assert optimized.improved and optimized.candidate.feasible and optimized.baseline.feasible
    design = optimized.candidate.design
    candidate_vehicle = sized_vehicle(vehicle, design, scale_dry_mass=settings.scale_dry_mass)
    candidate_guidance = replace(guidance, steering=replace(guidance.steering, pitch_kick_rad=design.pitch_kick_rad))
    verification = replace(config, max_step_s=.1, rtol=1e-10)
    candidate = trajectory_metrics(simulate(candidate_vehicle, verification, guidance=candidate_guidance),
                                    guidance.steering.rail_azimuth_rad)
    baseline = trajectory_metrics(simulate(vehicle, verification, guidance=guidance), guidance.steering.rail_azimuth_rad)
    assert candidate.apogee_altitude_m > baseline.apogee_altitude_m+1.
    assert candidate.max_q_pa <= settings.max_q_pa
    assert candidate.max_abs_axial_g <= settings.max_axial_g
    assert settings.stage1_fraction_bounds[0] <= design.stage1_propellant_fraction <= settings.stage1_fraction_bounds[1]
    assert settings.pitch_kick_bounds_rad[0] <= design.pitch_kick_rad <= settings.pitch_kick_bounds_rad[1]
    assert_allclose(candidate.apogee_altitude_m, optimized.candidate.objective_value, atol=.02, rtol=0.)


def test_budget_stop_is_not_optimizer_convergence() -> None:
    """A one-evaluation search cap must never masquerade as convergence."""
    vehicle, config, guidance = short_launch()
    result = optimize_trajectory(vehicle, config, guidance, OptimizationConfig(max_evaluations=1))
    assert not result.optimizer_converged
    assert len(result.evaluations) == 1
    assert "budget" in result.message.lower()
