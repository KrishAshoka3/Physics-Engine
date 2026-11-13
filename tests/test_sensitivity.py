"""Independent analytical variance benchmarks and sensitivity failure semantics."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from src.analysis import DispersionSpace, Uncertainty, sobol_analysis, standardized_regression
from src.analysis import launch_sobol
from tests.analysis_cases import short_launch


def test_standardized_linear_coefficients_and_correlated_caveat() -> None:
    """Orthogonal standardized predictors give exact beta^2 fractions 0.2 and 0.8."""
    x = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    y = x @ np.array([1., 2.])
    result = standardized_regression(x, y, ("a", "b"), ("altitude_m",), independent_inputs=True)
    assert_allclose(result.coefficients, [[1./np.sqrt(5.), 2./np.sqrt(5.)]])
    assert_allclose(result.linear_variance_indices, [[.2, .8]])
    assert_allclose(result.r_squared, 1.)
    assert standardized_regression(x, y, ("a", "b"), ("q_pa",)).linear_variance_indices is None
    zero = standardized_regression(x, np.zeros(4), ("a", "b"), ("constant",))
    assert zero.constant_outputs[0] and np.isnan(zero.r_squared[0])
    with pytest.raises(ValueError, match="Collinear"):
        standardized_regression(np.column_stack((x[:, 0], x[:, 0])), y, ("a", "b"), ("m",))


def test_sobol_ishigami_interaction_and_seed_reproducibility() -> None:
    """Ishigami first/total indices agree with its analytical variance decomposition within 0.025."""
    space = DispersionSpace(tuple(Uncertainty(f"x{i}", np.pi, "uniform") for i in range(3)))

    def ishigami(inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return dimensionless Ishigami output for independent uniform [-pi,pi] variables."""
        x1, x2, x3 = inputs.T
        return np.sin(x1)+7.*np.sin(x2)**2+.1*x3**4*np.sin(x1)

    result = sobol_analysis(ishigami, space, ("ishigami",), n_base=4096, seed=17, bootstrap_resamples=50)
    variance = .5+49./8.+.1*np.pi**4/5.+.01*np.pi**8/18.
    v1, v2 = .5*(1.+.1*np.pi**4/5.)**2, 49./8.
    interaction = variance-v1-v2
    assert_allclose(result.first_order, [[v1/variance, v2/variance, 0.]], atol=.025, rtol=0.)
    assert_allclose(result.total_order, [[(v1+interaction)/variance, v2/variance, interaction/variance]], atol=.025, rtol=0.)
    assert result.evaluation_count == 4096*5
    assert result.first_interval.shape == (1, 3, 2)
    again = sobol_analysis(ishigami, space, ("ishigami",), n_base=4096, seed=17, bootstrap_resamples=50)
    assert_allclose(result.first_order, again.first_order, rtol=0., atol=0.)
    assert_allclose(result.first_interval, again.first_interval, rtol=0., atol=0.)


def test_sobol_additive_convergence_and_invalid_outputs() -> None:
    """An additive independent model has matching first/total contributions; missing data cannot be dropped."""
    space = DispersionSpace((Uncertainty("a", 1., "uniform"), Uncertainty("b", 1., "uniform")))

    def additive(inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return additive dimensionless output plus a zero-variance diagnostic."""
        return np.column_stack((inputs[:, 0]+2.*inputs[:, 1], np.ones(len(inputs))))

    result = sobol_analysis(additive, space, ("linear", "constant"), n_base=1024, seed=3, bootstrap_resamples=30)
    assert_allclose(result.first_order[0], [.2, .8], atol=.005, rtol=0.)
    assert_allclose(result.half_sample_first[0], [.2, .8], atol=.01, rtol=0.)
    assert np.all(np.isnan(result.first_order[1]))
    with pytest.raises(ValueError, match="independent"):
        sobol_analysis(additive, replace(space, correlation=((1., .5), (.5, 1.))), ("linear", "constant"))

    def missing(inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return unavailable metrics to exercise the complete-design requirement."""
        return np.full(len(inputs), np.nan)

    with pytest.raises(ValueError, match="finite"):
        sobol_analysis(missing, space, ("missing",), n_base=4)


def test_launch_sobol_design_runs_physical_metrics() -> None:
    """Exercise the real-launch adapter and preserve its complete structured SI design."""
    vehicle, config, guidance = short_launch()
    space = DispersionSpace((Uncertainty("stage_1_propellant_fraction", .015),))
    result = launch_sobol(vehicle, config, guidance, space, n_base=8, seed=4, bootstrap_resamples=10)
    assert result.inputs.shape == (24, 1) and result.outputs.shape == (24, 3)
    assert np.all(np.isfinite(result.outputs))
    assert result.first_order.shape == (3, 1)
    with pytest.raises(ValueError, match="finite"):
        launch_sobol(vehicle, replace(config, duration_s=.1), guidance, space, n_base=4, bootstrap_resamples=2)
