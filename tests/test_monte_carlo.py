"""Distribution, covariance, impact projection, replay, and process-pool verification."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.analysis import (DispersionSpace, Uncertainty, covariance_ellipses, default_dispersion_space,
                          evaluate_design, impact_coordinates, perturb_launch, run_monte_carlo,
                          sample_inputs, trajectory_metrics, plot_dispersion)
from src.physics import coefficients, propulsion, simulate
from src.physics.constants import WGS84_A_M
from tests.analysis_cases import short_launch


def test_seeded_marginals_and_batch_length_independence() -> None:
    """Independent child streams reproduce prefixes and prescribed means/standard deviations."""
    space = default_dispersion_space()
    samples, seeds = sample_inputs(space, 10000, 7)
    prefix, prefix_seeds = sample_inputs(space, 4, 7)
    assert len(space.parameters) == 23
    assert_allclose(samples[:4], prefix, rtol=0., atol=0.)
    assert seeds[:4] == prefix_seeds
    for index, parameter in enumerate(space.parameters):
        expected_std = parameter.half_range/(3. if parameter.distribution == "gaussian" else np.sqrt(3.))
        assert abs(samples[:, index].mean()) < .04*expected_std
        assert_allclose(samples[:, index].std(), expected_std, rtol=.035)
        if parameter.distribution == "uniform":
            assert np.max(np.abs(samples[:, index])) <= parameter.half_range


def test_correlated_gaussian_inputs() -> None:
    """A specified latent Gaussian correlation must reproduce Pearson rho within 0.02."""
    space = DispersionSpace((Uncertainty("a", 1.), Uncertainty("b", 1.)), correlation=((1., .7), (.7, 1.)))
    samples, _ = sample_inputs(space, 10000, 9)
    assert abs(np.corrcoef(samples.T)[0, 1]-.7) < .02
    with pytest.raises(ValueError, match="positive definite"):
        replace(space, correlation=((1., 1.), (1., 1.)))


@pytest.mark.parametrize("curve", [False, True])
def test_physical_perturbations_independent_thrust_isp_and_full_drag(curve: bool) -> None:
    """Thrust/Isp perturb independently, all Cd terms scale, and nominal inputs stay immutable."""
    vehicle, config, guidance = short_launch()
    if curve:
        stage = replace(vehicle.stages[0], mass_flow_kg_s=None, thrust_curve_time_s=(0., 1.), thrust_curve_vac_n=(600., 600.))
        vehicle = replace(vehicle, stages=(stage, vehicle.stages[1]))
    space = DispersionSpace((Uncertainty("stage_1_thrust_fraction", .03), Uncertainty("stage_1_isp_fraction", .015),
                             Uncertainty("cd_fraction", .1), Uncertainty("elevation_rad", .01, unit="rad")))
    changed, _, program = perturb_launch(vehicle, config, guidance, space, [.03, .015, .1, .005])
    nominal = propulsion(vehicle.stages[0], .5, 0.)
    perturbed = propulsion(changed.stages[0], .5, 0.)
    assert_allclose(perturbed.vacuum_thrust_n/nominal.vacuum_thrust_n, 1.03)
    assert_allclose(perturbed.mass_flow_kg_s/nominal.mass_flow_kg_s, 1.03/1.015)
    mach, alpha_rad = np.array([.5, 1., 5.]), np.array([.2, .6, 1.])
    nominal_cd, nominal_cl = coefficients(mach, alpha_rad, vehicle.stages[0].aero)
    changed_cd, changed_cl = coefficients(mach, alpha_rad, changed.stages[0].aero)
    assert_allclose(changed_cd, 1.1*nominal_cd)
    assert_allclose(changed_cl, nominal_cl)
    assert vehicle.stages[0].aero.cd_multiplier == 1.
    assert program.steering.rail_elevation_rad > np.pi/2.
    assert program.steering.rail_azimuth_rad == guidance.steering.rail_azimuth_rad


def test_wind_superposition_preserves_nominal_knots() -> None:
    """Add a sheared east gust in m/s while retaining nominal wind interpolation breakpoints."""
    from src.physics import WindProfile
    vehicle, config, guidance = short_launch()
    config = replace(config, wind=WindProfile((0., 5000., 9000.), ((1., 2., 3.), (8., 2., 3.), (1., 2., 3.))))
    space = DispersionSpace((Uncertainty("wind_0_e_m_s", 15., unit="m/s"),
                             Uncertainty("wind_1_e_m_s", 15., unit="m/s")), wind_altitude_m=(0., 10000.))
    _, changed, _ = perturb_launch(vehicle, config, guidance, space, [0., 10.])
    heights_m = np.array([0., 2000., 5000., 9000., 10000.])
    assert_allclose(changed.wind.velocity(heights_m)-config.wind.velocity(heights_m),
                    np.column_stack((heights_m/1000., np.zeros((len(heights_m), 2)))))


def test_covariance_ellipse_geometry_and_joint_coverage() -> None:
    """Every contour vertex has exact sigma-squared Mahalanobis distance within 1e-12."""
    angle_rad = .4
    rotation = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
    points_m = np.array([[3., 0.], [-3., 0.], [0., 1.], [0., -1.]]) @ rotation.T + [100., -20.]
    ellipses = covariance_ellipses(points_m)
    expected_m2 = rotation @ np.diag([6., 2./3.]) @ rotation.T
    for ellipse in ellipses:
        assert_allclose(ellipse.covariance_m2, expected_m2, atol=1e-13)
        centered = ellipse.points_m-ellipse.center_m
        distance_sq = np.einsum("ni,ij,nj->n", centered, np.linalg.inv(expected_m2), centered)
        assert_allclose(distance_sq, ellipse.sigma**2, rtol=0., atol=1e-12)
        assert_allclose(ellipse.gaussian_coverage, 1.-np.exp(-ellipse.sigma**2/2.))
    assert_allclose(ellipses[0].semiaxes_m, np.sqrt([6., 2./3.]))
    assert ellipses[0].gaussian_coverage < .4
    degenerate = covariance_ellipses([[0., 0.], [1., 0.], [2., 0.]])[0]
    assert degenerate.rank == 1 and np.isnan(degenerate.gaussian_coverage)
    with pytest.raises(ValueError):
        covariance_ellipses([[np.nan, 0.], [0., 1.]])


def test_impact_projection_known_arcs_and_antimeridian() -> None:
    """Equatorial small arcs have exact projected length in the declared spherical map."""
    assert_allclose(impact_coordinates(0., .01, 0., 0., np.pi/2.), [WGS84_A_M*.01, 0.], atol=1e-9)
    assert_allclose(impact_coordinates(0., -np.pi+.005, 0., np.pi-.005, np.pi/2.), [WGS84_A_M*.01, 0.], atol=1e-7)
    assert_allclose(impact_coordinates(.01, 0., 0., 0., np.pi/2.), [0., -WGS84_A_M*.01], atol=1e-9)
    with pytest.raises(ValueError, match="antipode"):
        impact_coordinates(0., np.pi, 0., 0., 0.)


def test_parallel_serial_replay_failures_and_json(tmp_path: Path) -> None:
    """Spawned workers reproduce serial trajectories and preserve invalid realizations in place."""
    vehicle, config, guidance = short_launch()
    space = DispersionSpace((Uncertainty("stage_1_dry_fraction", .02),))
    inputs = np.array([[0.], [.01], [-2.], [-.01]])
    serial = evaluate_design(vehicle, config, guidance, space, inputs)
    parallel = evaluate_design(vehicle, config, guidance, space, inputs, workers=2)
    assert [run.status for run in serial] == [run.status for run in parallel]
    assert serial[2].status == "failed" and "ValueError" in serial[2].error
    for first, second in zip(serial, parallel):
        assert first.index == second.index
        if first.metrics is not None:
            assert_allclose(first.metrics.apogee_altitude_m, second.metrics.apogee_altitude_m, rtol=0., atol=0.)
            assert_allclose(first.metrics.downrange_impact_m, second.metrics.downrange_impact_m, rtol=0., atol=0.)
    result = run_monte_carlo(vehicle, config, guidance, count=4, seed=6, workers=1, space=space)
    assert result.metadata["status_counts"] == dict(impact=4, horizon=0, failed=0)
    result.save(tmp_path/"ensemble.json")
    plot_dispersion(result, tmp_path/"ellipse.png")
    assert (tmp_path/"ellipse.png").read_bytes().startswith(b"\x89PNG")
    saved = json.loads((tmp_path/"ensemble.json").read_text())
    assert saved["runs"][0]["metrics"]["insertion_radius_error_m"] is None
    assert saved["configuration"]["space"]["parameters"][0]["name"] == space.names[0]
    short = run_monte_carlo(vehicle, replace(config, duration_s=.1), guidance, count=2, workers=1, space=space)
    assert not short.ellipses
    assert short.statistics["apogee_altitude_m"]["count"] == 0
    assert short.statistics["downrange_impact_m"]["missing"] == 2


def test_refined_apogee_and_peak_load_step_convergence() -> None:
    """Halving the sampling step must preserve refined apogee within 0.02 m and Q within 1%."""
    vehicle, config, guidance = short_launch()
    first = trajectory_metrics(simulate(vehicle, config, guidance=guidance), np.pi/2.)
    second = trajectory_metrics(simulate(vehicle, replace(config, max_step_s=.25), guidance=guidance), np.pi/2.)
    assert_allclose(first.apogee_altitude_m, second.apogee_altitude_m, atol=.02, rtol=0.)
    assert_allclose(first.max_q_pa, second.max_q_pa, rtol=.01)


def test_burnout_left_side_axial_load_is_included() -> None:
    """The exact pre-cutoff thrust/mass peak must survive right-continuous output stitching."""
    vehicle, config, guidance = short_launch()
    config = replace(config, atmosphere_enabled=False)
    result = simulate(vehicle, config, guidance=guidance)
    cutoff = next(event for event in result.events if event.name == "SECO")
    stage = vehicle.stages[1]
    expected_g = stage.mass_flow_kg_s*stage.isp_vac_s/cutoff.state_before[6]
    assert_allclose(cutoff.telemetry_before["axial_g"], expected_g, atol=1e-10)
    metrics = trajectory_metrics(result, guidance.steering.rail_azimuth_rad)
    assert metrics.max_abs_axial_g >= expected_g-1e-10
    index = np.argmin(abs(result.time_s-cutoff.time_s))
    assert result.telemetry["thrust_n"][index] == 0.


def test_ground_event_survives_trial_below_atmosphere_domain() -> None:
    """A fast vacuum-like drop locates ground within 1e-5 s despite deep underground trial stages."""
    vehicle, config, _ = short_launch()
    vehicle = replace(vehicle, stages=tuple(replace(stage, reference_area_m2=0.) for stage in vehicle.stages))
    config = replace(config, launch_altitude_m=10000., initial_velocity_enu_m_s=(0., 0., -2500.),
                     engines_enabled=False, max_step_s=8., rtol=1e-7)
    coarse = simulate(vehicle, config)
    fine = simulate(vehicle, replace(config, max_step_s=.1, rtol=1e-10))
    assert coarse.termination == fine.termination == "ground_impact"
    assert_allclose(coarse.time_s[-1], fine.time_s[-1], atol=1e-5, rtol=0.)
    assert_allclose(coarse.telemetry["altitude_m"][-1], 0., atol=1e-6)
