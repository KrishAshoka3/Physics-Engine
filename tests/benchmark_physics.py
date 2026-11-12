"""Reproducible Phase 1 audit measurements; run ``python -m tests.benchmark_physics``.

Notes
-----
Writes JSON to stdout, not into the repository. Timings [s] are machine-specific
observations, not flaky CI assertions. All trajectory inputs are deterministic.
"""

import cProfile
import io
import json
import platform
import pstats
from time import perf_counter

import numpy as np
import scipy

from src.physics import PRESETS, SimulationConfig, StageConfig, VehicleConfig, atmosphere, simulate
from src.physics.constants import EARTH_MU_M3_S2, EARTH_OMEGA_RAD_S, G0_M_S2, WGS84_A_M


def main() -> None:
    """Print JSON with SI benchmark errors, timings [s], versions, and top profile costs."""
    vehicle = next(iter(PRESETS.values()))
    profiler = cProfile.Profile()
    started_s = perf_counter()
    profiler.enable()
    sounding = simulate(vehicle, SimulationConfig(duration_s=700.))
    profiler.disable()
    simulation_elapsed_s = perf_counter() - started_s
    profile_text = io.StringIO()
    pstats.Stats(profiler, stream=profile_text).strip_dirs().sort_stats("cumtime").print_stats(15)

    heights_m = np.linspace(0., 500000., 10000)
    # Warm up both paths; report best-of-five batch time and one scalar-loop time.
    atmosphere(heights_m)
    batch_times_s: list[float] = []
    for _ in range(5):
        started_s = perf_counter()
        atmosphere(heights_m)
        batch_times_s.append(perf_counter() - started_s)
    started_s = perf_counter()
    scalar_pressure_pa = np.array([float(atmosphere(height_m).pressure_pa) for height_m in heights_m])
    scalar_elapsed_s = perf_counter() - started_s
    batch_pressure_pa = atmosphere(heights_m).pressure_pa

    test_vehicle = VehicleConfig("benchmark", (StageConfig("test", 10., 100., 300., 0., mass_flow_kg_s=10.),), 10.)
    tsiolkovsky_errors_m_s: dict[str, float] = {}
    for method in ("DOP853", "RK45"):
        result = simulate(test_vehicle, SimulationConfig(duration_s=12., launch_altitude_m=100.,
                          atmosphere_enabled=False, rotating_earth=False, mu_m3_s2=0., method=method))
        tsiolkovsky_errors_m_s[method] = abs(result.state[-1, 3] - 300.*G0_M_S2*np.log(6.))

    radius_m = WGS84_A_M + 400000.
    orbital_speed_m_s = np.sqrt(EARTH_MU_M3_S2/radius_m)
    orbital = simulate(test_vehicle, SimulationConfig(
        duration_s=2.*np.pi*radius_m/orbital_speed_m_s, launch_altitude_m=400000., atmosphere_enabled=False,
        engines_enabled=False, j2_enabled=False, max_step_s=60.,
        initial_velocity_enu_m_s=(orbital_speed_m_s - EARTH_OMEGA_RAD_S*radius_m, 0., 0.)))
    energy_m2_s2 = orbital.telemetry["specific_energy_m2_s2"]
    record: dict[str, object] = {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "sounding_profiled_runtime_s": simulation_elapsed_s,
        "sounding_function_evaluations": sounding.function_evaluations,
        "sounding_samples": len(sounding.time_s),
        "sounding_sampled_apogee_m": float(sounding.telemetry["altitude_m"].max()),
        "sounding_impact_time_s": float(sounding.time_s[-1]),
        "sounding_event_times_s": {event.name: event.time_s for event in sounding.events},
        "atmosphere_batch_10000_best_s": min(batch_times_s),
        "atmosphere_scalar_loop_10000_s": scalar_elapsed_s,
        "atmosphere_batch_speedup": scalar_elapsed_s/min(batch_times_s),
        "atmosphere_batch_scalar_max_pressure_difference_pa": float(np.max(np.abs(batch_pressure_pa - scalar_pressure_pa))),
        "tsiolkovsky_error_m_s": tsiolkovsky_errors_m_s,
        "one_orbit_max_relative_energy_error": float(np.max(np.abs(energy_m2_s2/energy_m2_s2[0] - 1.))),
        "one_orbit_position_closure_error_m": float(np.linalg.norm(orbital.state[-1, :3]-orbital.state[0, :3])),
        "profile": profile_text.getvalue(),
    }
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
