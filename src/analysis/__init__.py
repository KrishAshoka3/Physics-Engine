"""Phase 3 uncertainty, sensitivity, and constrained trajectory sizing APIs."""

from .metrics import TrajectoryMetrics, impact_coordinates, trajectory_metrics, write_json
from .monte_carlo import (DispersionEllipse, DispersionSpace, MonteCarloResult, Uncertainty,
                          covariance_ellipses, default_dispersion_space, evaluate_design,
                          perturb_launch, plot_dispersion, run_monte_carlo, sample_inputs)
from .optimizer import (DesignPoint, OptimizationConfig, OptimizationResult,
                         evaluate_candidate, optimize_trajectory, sized_vehicle)
from .sensitivity import (RegressionSensitivity, SobolSensitivity, launch_sobol,
                           regression_from_ensemble, sobol_analysis, standardized_regression)

__all__ = ["TrajectoryMetrics", "impact_coordinates", "trajectory_metrics", "write_json",
           "DispersionEllipse", "DispersionSpace", "MonteCarloResult", "Uncertainty", "covariance_ellipses",
           "default_dispersion_space", "evaluate_design", "perturb_launch", "plot_dispersion",
           "run_monte_carlo", "sample_inputs", "DesignPoint", "OptimizationConfig", "OptimizationResult",
           "evaluate_candidate", "optimize_trajectory", "sized_vehicle", "RegressionSensitivity",
           "SobolSensitivity", "launch_sobol", "regression_from_ensemble", "sobol_analysis", "standardized_regression"]
