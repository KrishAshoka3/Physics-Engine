"""Deterministic Phase 2 audit; run ``python -m tests.benchmark_guidance``.

Notes
-----
Prints JSON to stdout with SI measurements, versions, base Git revision, and
profiling evidence. Timing is machine-specific and is not a CI acceptance limit.
The committed record belongs to the milestone containing this file, including
any working-tree changes recorded at benchmark time. No randomness is used.
"""

import cProfile
import io
import json
import platform
import pstats
import subprocess
from time import perf_counter

import numpy as np
import scipy

from src.guidance import GuidanceProgram, OrbitTarget, SteeringConfig
from src.physics import PRESETS, SimulationConfig, SimulationResult, simulate
from tests.guidance_cases import wind_tracking_case


def measurements(result: SimulationResult, elapsed_s: float) -> dict[str, object]:
    """Collect SI results and dimensionless counts/flags from an accepted trajectory."""
    plans = [row for row in result.guidance_records if row["phase"] == "orbital_plan"]
    record: dict[str, object] = {
        "runtime_s": elapsed_s,
        "function_evaluations": result.function_evaluations,
        "samples": len(result.time_s),
        "termination": result.termination,
        "event_times_s": {event.name: event.time_s for event in result.events},
        "insertion": [row for row in result.guidance_records if row["phase"] == "insertion"],
        "orbital_plans": len(plans),
        "feasible_plans": sum(bool(row["feasible"]) for row in plans),
    }
    if result.config.dof == 6:
        powered = result.engine_on & (result.time_s > 20.)
        gimbal_rad = result.telemetry["gimbal_rad"]
        slew_rad_s = np.diff(gimbal_rad, axis=0)/np.diff(result.time_s)[:, None]
        record.update({
            "max_gimbal_rad": float(np.max(np.abs(gimbal_rad))),
            "max_gimbal_rate_rad_s": float(np.max(np.abs(slew_rad_s))),
            "max_powered_attitude_error_after_20_s_rad": float(np.max(result.telemetry["attitude_error_rad"][powered])),
            "final_attitude_error_rad": float(result.telemetry["attitude_error_rad"][-1]),
            "max_quaternion_norm_error": float(np.max(np.abs(result.telemetry["quaternion_norm"]-1.))),
        })
    return record


def main() -> None:
    """Print repeatable orbital/gust measurements and cumulative profile costs as JSON."""
    cases: dict[str, object] = {}
    profile = cProfile.Profile()
    for mode in ("PID", "LQR"):
        vehicle, config, program = wind_tracking_case(mode)
        started_s = perf_counter()
        if mode == "PID":
            profile.enable()
        result = simulate(vehicle, config, guidance=program)
        profile.disable()
        cases[f"wind_shear_{mode}"] = measurements(result, perf_counter()-started_s)
    profile_text = io.StringIO()
    pstats.Stats(profile, stream=profile_text).strip_dirs().sort_stats("cumtime").print_stats(18)
    missions = (("Small-Lift Orbital Vehicle (Electron Class)", 3, 425.),
                ("Medium-Lift Two-Stage Orbital (Falcon 9 Class)", 3, 460.),
                ("Small-Lift Orbital Vehicle (Electron Class)", 6, 425.))
    for name, dof, duration_s in missions:
        started_s = perf_counter()
        result = simulate(PRESETS[name], SimulationConfig(dof=dof, duration_s=duration_s, max_step_s=5.),
                          guidance=GuidanceProgram(SteeringConfig(target=OrbitTarget())))
        cases[f"{name}_{dof}DoF"] = measurements(result, perf_counter()-started_s)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.strip())
    print(json.dumps({"base_git_revision": revision, "working_tree_dirty": dirty,
                      "python": platform.python_version(), "platform": platform.platform(),
                      "numpy": np.__version__, "scipy": scipy.__version__,
                      "cases": cases, "wind_shear_PID_profile": profile_text.getvalue()}, indent=2))


if __name__ == "__main__":
    main()
