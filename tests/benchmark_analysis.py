"""Reproducible Phase 3 launch ensemble, scalability, sensitivity, and sizing audit.

Notes
-----
Run ``python -m tests.benchmark_analysis`` from the repository root. The main
guard is required by Windows spawn. Writes JSON and a dispersion figure into
docs; progress is printed while the workload runs. No stochastic flight data
or empirical vehicle calibration is implied by these engineering scenarios.
"""

from dataclasses import asdict
import cProfile
import io
from pathlib import Path
import pstats
from time import perf_counter

import numpy as np

from src.analysis import (OptimizationConfig, optimize_trajectory, plot_dispersion,
                          regression_from_ensemble, run_monte_carlo, trajectory_metrics, write_json)
from src.guidance import GuidanceProgram
from src.physics import PRESETS, SimulationConfig, simulate
from tests.analysis_cases import short_launch


def main() -> None:
    """Record SI results, workload durations [s], counts, and measured profile costs."""
    output = Path(__file__).resolve().parents[1]/"docs"
    vehicle = next(iter(PRESETS.values()))
    config = SimulationConfig(duration_s=650., max_step_s=4., rtol=1e-7)
    guidance = GuidanceProgram()
    profile = cProfile.Profile()
    started_s = perf_counter()
    profile.enable()
    nominal = trajectory_metrics(simulate(vehicle, config, guidance=guidance), guidance.steering.rail_azimuth_rad)
    profile.disable()
    nominal_runtime_s = perf_counter()-started_s
    text = io.StringIO()
    pstats.Stats(profile, stream=text).strip_dirs().sort_stats("cumtime").print_stats(15)
    print("Nominal sounding profile complete", flush=True)

    started_s = perf_counter()
    ensemble = run_monte_carlo(vehicle, config, guidance, count=100, seed=20261112, workers=4)
    ensemble_runtime_s = perf_counter()-started_s
    ensemble.save(output/"phase3_ensemble.json")
    plot_dispersion(ensemble, output/"phase3_dispersion.png")
    print(f"100 sounding launches: {ensemble.metadata['status_counts']}", flush=True)
    regression = regression_from_ensemble(ensemble)

    test_vehicle, test_config, test_guidance = short_launch()
    started_s = perf_counter()
    scaling = run_monte_carlo(test_vehicle, test_config, test_guidance, count=1000, seed=37, workers=4)
    scaling_runtime_s = perf_counter()-started_s
    print(f"1000 short fixture launches: {scaling.metadata['status_counts']}", flush=True)

    settings = OptimizationConfig(stage1_fraction_bounds=(.3, .65), pitch_kick_bounds_rad=(0., np.deg2rad(8.)),
                                   max_q_pa=nominal.max_q_pa*1.05, max_axial_g=nominal.max_abs_axial_g*1.05,
                                   max_iterations=5, max_evaluations=30)
    started_s = perf_counter()
    optimization = optimize_trajectory(vehicle, config, guidance, settings)
    optimization_runtime_s = perf_counter()-started_s
    write_json(output/"phase3_optimization.json", optimization)
    print(f"Sounding optimization: feasible={optimization.candidate.feasible}, improved={optimization.improved}", flush=True)
    write_json(output/"phase3_benchmark.json", dict(
        metadata=ensemble.metadata, nominal=asdict(nominal), nominal_profiled_runtime_s=nominal_runtime_s,
        nominal_profile=text.getvalue(), ensemble_runtime_s=ensemble_runtime_s,
        ensemble_count=100, regression=regression, short_fixture_count=1000,
        short_fixture_runtime_s=scaling_runtime_s, short_fixture_status_counts=scaling.metadata["status_counts"],
        optimization_runtime_s=optimization_runtime_s, optimization_feasible=optimization.candidate.feasible,
        optimization_improved=optimization.improved))


if __name__ == "__main__":
    main()
