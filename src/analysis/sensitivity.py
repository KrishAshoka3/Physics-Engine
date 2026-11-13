"""Standardized regression screening and independent-input Saltelli Sobol indices.

Notes
-----
Regression coefficient squares are linear-model approximations, not general
Sobol indices. True first/total-order indices use a structured pick-freeze
design and SciPy's Saltelli estimator. Invalid/missing outputs are rejected;
dropping failed launches would destroy the sampling design and bias results.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import qmc, sobol_indices

from src.guidance import GuidanceProgram
from src.physics import SimulationConfig, VehicleConfig
from .monte_carlo import DispersionSpace, MonteCarloResult, evaluate_design

FloatArray = NDArray[np.float64]
BatchEvaluator = Callable[[FloatArray], FloatArray]


@dataclass(frozen=True)
class RegressionSensitivity:
    """Dimensionless standardized coefficients (output,input), fit R^2, and rank.

    Attributes
    ----------
    linear_variance_indices : ndarray or None
        beta^2 only when independent_inputs=True. Valid as variance attribution
        only for a sufficiently additive linear response; never renormalized.
    input_correlation : ndarray
        Empirical Pearson matrix to reveal finite-sample correlations.
    constant_outputs : ndarray of bool
        Outputs with zero variance have NaN coefficients/indices/R^2.
    """

    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    coefficients: FloatArray
    linear_variance_indices: FloatArray | None
    r_squared: FloatArray
    input_correlation: FloatArray
    condition_number: float
    constant_outputs: NDArray[np.bool_]


def standardized_regression(inputs: ArrayLike, outputs: ArrayLike, input_names: tuple[str, ...],
                            output_names: tuple[str, ...], *, independent_inputs: bool = False) -> RegressionSensitivity:
    """Fit standardized OLS with intercept removed by centering.

    Parameters
    ----------
    inputs, outputs : array_like
        Shape (n,d) physical perturbations and (n,s) SI metrics. Units are
        specified in the supplied column names; coefficients are dimensionless.
    input_names, output_names : tuple of str
        Ordered unique column identifiers.
    independent_inputs : bool
        User/model declaration permitting beta-squared linear contributions.
        Correlated designs still support coefficients but not this attribution.

    Raises
    ------
    ValueError
        Missing data, insufficient observations, constant/collinear inputs.
    """
    x, y = np.asarray(inputs, dtype=float), np.asarray(outputs, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if (x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or x.shape[1] != len(input_names)
            or y.shape[1] != len(output_names) or len(x) <= x.shape[1]+1
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))):
        raise ValueError("Require finite aligned data, named columns, and n > d+1")
    if len(set(input_names)) != len(input_names) or len(set(output_names)) != len(output_names):
        raise ValueError("Column names must be unique")
    x_scale = x.std(axis=0, ddof=1)
    if np.any(x_scale == 0.):
        raise ValueError("Constant input columns cannot be standardized")
    standardized_x = (x-x.mean(axis=0))/x_scale
    y_scale = y.std(axis=0, ddof=1)
    constant = y_scale == 0.
    standardized_y = (y-y.mean(axis=0))/np.where(constant, 1., y_scale)
    beta, _, rank, singular = np.linalg.lstsq(standardized_x, standardized_y, rcond=None)
    if rank != x.shape[1]:
        raise ValueError("Collinear inputs do not identify individual regression coefficients")
    residual = standardized_y-standardized_x @ beta
    r_squared = 1.-np.sum(residual**2, axis=0)/(len(x)-1)
    beta[:, constant], r_squared[constant] = np.nan, np.nan
    return RegressionSensitivity(input_names, output_names, beta.T, beta.T**2 if independent_inputs else None,
                                  r_squared, np.atleast_2d(np.corrcoef(standardized_x, rowvar=False)),
                                  float(singular[0]/singular[-1]), constant)


def regression_from_ensemble(result: MonteCarloResult,
                             metrics: tuple[str, ...] = ("apogee_altitude_m", "max_q_pa", "downrange_impact_m")) -> RegressionSensitivity:
    """Screen an ensemble without silently dropping failed/censored metric rows.

    Notes
    -----
    Select available metrics explicitly for orbital missions without impacts.
    Missing outcomes require a separate failure-probability analysis.
    """
    if any(run.metrics is None for run in result.runs):
        raise ValueError("Failed runs present; conditional regression must be an explicit separate analysis")
    outputs = np.array([[getattr(run.metrics, name) for name in metrics] for run in result.runs])
    correlation = result.configuration["space"]["correlation"]
    independent = correlation is None or np.array_equal(correlation, np.eye(len(result.input_names)))
    return standardized_regression(result.inputs, outputs, result.input_names, metrics, independent_inputs=independent)


@dataclass(frozen=True)
class SobolSensitivity:
    """First/total indices and 95% bootstrap intervals, all dimensionless.

    Notes
    -----
    Indices have shape (outputs,inputs), intervals append a (low,high) axis.
    Half-sample estimates expose convergence; raw negative/>1 Monte Carlo
    estimates are retained. Zero output variance produces NaNs, not fake ranks.
    inputs/outputs retain the structured design for audit/replay in original SI
    units. seed, n_base, and evaluation_count are dimensionless metadata.
    """

    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    first_order: FloatArray
    total_order: FloatArray
    first_interval: FloatArray
    total_interval: FloatArray
    half_sample_first: FloatArray
    constant_outputs: NDArray[np.bool_]
    seed: int
    n_base: int
    evaluation_count: int
    inputs: FloatArray
    outputs: FloatArray


def sobol_analysis(evaluator: BatchEvaluator, space: DispersionSpace, output_names: tuple[str, ...], *,
                    n_base: int = 256, seed: int = 0, bootstrap_resamples: int = 200) -> SobolSensitivity:
    """Evaluate a scrambled Sobol pick-freeze design using n_base*(d+2) model runs.

    Parameters
    ----------
    evaluator : callable
        Accept (n,d) physical perturbations, return finite (n,s) SI outputs.
    space : DispersionSpace
        Independent marginal uncertainty distributions; correlated inputs rejected.
    output_names : tuple of str
        Ordered metric identifiers carrying SI units.
    n_base, seed, bootstrap_resamples : int
        Power-of-two base count >=4, seed, and bootstrap count >=2. Increase
        n_base and compare half-sample estimates/intervals before interpreting.
    """
    if not isinstance(n_base, int) or n_base < 4 or n_base & (n_base-1):
        raise ValueError("n_base must be a power of two >=4")
    if seed < 0 or bootstrap_resamples < 2 or not output_names:
        raise ValueError("Require nonnegative seed, >=2 resamples, and named outputs")
    dimension = len(space.parameters)
    if space.correlation is not None and not np.array_equal(space.correlation, np.eye(dimension)):
        raise ValueError("Classical Sobol variance decomposition requires independent inputs")
    unit = qmc.Sobol(d=2*dimension, scramble=True, seed=seed).random_base2(int(np.log2(n_base)))
    unit = np.clip(unit, np.finfo(float).eps, 1.-np.finfo(float).eps)
    a, b = space.transform(unit[:, :dimension]), space.transform(unit[:, dimension:])
    hybrids = np.broadcast_to(a, (dimension, *a.shape)).copy()
    for index in range(dimension):
        hybrids[index, :, index] = b[:, index]
    inputs = np.concatenate((a, b, hybrids.reshape(-1, dimension)))
    outputs = np.asarray(evaluator(inputs), dtype=float)
    if outputs.ndim == 1:
        outputs = outputs[:, None]
    if outputs.shape != (len(inputs), len(output_names)) or not np.all(np.isfinite(outputs)):
        raise ValueError("Every Sobol model evaluation must return finite named outputs; no row dropping")
    fa, fb = outputs[:n_base].T, outputs[n_base:2*n_base].T
    fab = outputs[2*n_base:].reshape(dimension, n_base, -1).transpose(0, 2, 1)
    constant = np.var(np.concatenate((fa, fb), axis=1), axis=1) == 0.
    active = ~constant
    first = np.full((len(output_names), dimension), np.nan)
    total, half = first.copy(), first.copy()
    first_ci, total_ci = np.full((*first.shape, 2), np.nan), np.full((*first.shape, 2), np.nan)
    if np.any(active):
        evaluations = dict(f_A=fa[active], f_B=fb[active], f_AB=fab[:, active, :])
        result = sobol_indices(func=evaluations, n=n_base, rng=np.random.default_rng(seed))
        shape = (int(np.sum(active)), dimension)
        first[active], total[active] = np.reshape(result.first_order, shape), np.reshape(result.total_order, shape)
        # SciPy SobolResult.bootstrap does not accept/retain an RNG in the
        # installed API. Resample paired design columns explicitly for replay.
        generator = np.random.default_rng(seed)
        bootstrap_first, bootstrap_total = [], []
        for _ in range(bootstrap_resamples):
            indices = generator.integers(0, n_base, n_base)
            resampled = {key: value[..., indices] for key, value in evaluations.items()}
            sample = sobol_indices(func=resampled, n=n_base)
            first_sample, total_sample = np.reshape(sample.first_order, shape), np.reshape(sample.total_order, shape)
            zero_variance = np.var(np.concatenate((resampled["f_A"], resampled["f_B"]), axis=1), axis=1) == 0.
            first_sample[zero_variance], total_sample[zero_variance] = np.nan, np.nan
            bootstrap_first.append(first_sample)
            bootstrap_total.append(total_sample)
        first_ci[active] = np.moveaxis(np.nanquantile(bootstrap_first, [.025, .975], axis=0), 0, -1)
        total_ci[active] = np.moveaxis(np.nanquantile(bootstrap_total, [.025, .975], axis=0), 0, -1)
        half_result = sobol_indices(func={key: value[..., :n_base//2] for key, value in evaluations.items()},
                                    n=n_base//2, rng=np.random.default_rng(seed))
        half[active] = np.reshape(half_result.first_order, shape)
    return SobolSensitivity(space.names, output_names, first, total, first_ci, total_ci, half, constant,
                             seed, n_base, len(inputs), inputs, outputs)


def launch_sobol(vehicle: VehicleConfig, config: SimulationConfig, guidance: GuidanceProgram,
                 space: DispersionSpace, *, metrics: tuple[str, ...] = ("apogee_altitude_m", "max_q_pa", "downrange_impact_m"),
                 n_base: int = 256, seed: int = 0, workers: int = 1,
                 bootstrap_resamples: int = 200) -> SobolSensitivity:
    """Run physical launch Sobol analysis through the shared process-pool evaluator.

    Notes
    -----
    Default 23-dimensional uncertainties require 6400 runs at n_base=256;
    screen with regression or select a smaller justified subspace first.
    The caller must protect parallel execution with a Windows __main__ guard.
    """
    def evaluate(inputs: FloatArray) -> FloatArray:
        """Evaluate SI metrics for the complete structured design, rejecting missing runs."""
        records = evaluate_design(vehicle, config, guidance, space, inputs, workers=workers)
        failed = [record.index for record in records if record.metrics is None]
        if failed:
            raise ValueError(f"Sobol design has failed launches at indices {failed[:10]}")
        return np.array([[getattr(record.metrics, name) for name in metrics] for record in records])
    return sobol_analysis(evaluate, space, metrics, n_base=n_base, seed=seed, bootstrap_resamples=bootstrap_resamples)
