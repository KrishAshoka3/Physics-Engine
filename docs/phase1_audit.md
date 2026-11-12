# Phase 1 implementation audit

Audit date: 2026-11-12. This record covers the core modules requested for Phase 1, not flight-data validation or a completed guidance system. The associated milestone is the Git commit containing this report and `phase1_benchmark.json` (inspect with `git log -1 -- docs/phase1_audit.md`).

## Commands and outcomes

Follow-up completion audit: `tests/test_physics.py` adds 18 consolidated acceptance cases. The full suite now passes **68 tests** (4.96 s in the completion run). Both DOP853 and RK45 improve against analytical delta-V when changing `rtol` from 1e-6 to 1e-9. Mass budgets include dry mass, residual fuel, consumed exhaust, fairing, and retained payload/stage in both 3DoF and 6DoF. No implementation fixes were necessary. The original 50-case measurements below remain the initial implementation audit.

Atmosphere acceptance uses fixed published reference values with explicit rounding tolerances, not bitwise equality: 0/11/20/47 km are geopotential layer heights; 86 km is geometric with the US76 molecular-weight correction. The 86 km check is identified as US76 rather than falsely attributed to an independently inspected ICAO table. Reference provenance is in the test module and model notes.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m tests.benchmark_physics
git diff --check
```

- **50 pytest cases passed** in the final numerical test run (about 4 s locally).
- Compilation passed. An AST audit confirmed argument/return type annotations and docstrings on every defined function in the physics and test modules. This is not a claim of a full static type-checker proof.
- All eight requested dependencies remain installed in `.venv`. No new requirement was needed for the implementation. The audit used Python 3.12.14, NumPy 2.5.4, SciPy 1.18.1, and pytest 9.2.0 on Windows 11.
- No phase-2 guidance/autopilot, Monte Carlo analysis, UI, or PDF generator was implemented.

## Acceptance criteria and evidence

Thresholds are explicit assertions in the test files. They were defined before accepting benchmark results; implementation defects were corrected rather than weakening failed tests.

| Check | Acceptance criterion | Result |
| --- | --- | --- |
| US76 geopotential layer tables | Pressure relative error <= 2e-4; temperature absolute error <= 0.001 K | Pass |
| Independent geometric table points | Pressure/density relative error <= 1e-3; temperature error <= 0.003 K | Pass |
| Molecular correction | Table 8 ratios at 80/83/86 km and consistent sound speed/ideal gas relation | Pass |
| 86 km extension | Matched one-sided derivatives within 2e-4 relative; positive finite T/P/rho/a through 500 km | Pass |
| Atmosphere batch versus scalar | Numerically equivalent results | Maximum pressure difference 1.46e-11 Pa |
| Tsiolkovsky, DOP853 | Delta-V absolute error < 2e-5 m/s | 5.50e-7 m/s |
| Tsiolkovsky, RK45 | Delta-V absolute error < 2e-5 m/s | 2.63e-6 m/s |
| Cutoff/depletion/separation/ignition timing | Analytic event epochs within 1e-8 s | Pass |
| Fairing altitude and impact timing | Constant-speed analytic roots within 1e-6 s | Pass |
| Stage/fairing mass accounting | Correct dry/residual-fuel drops within 1e-7 kg; no position/velocity jumps | Pass |
| ECI/ENU powered equivalence | Position within 0.002 m, velocity within 2e-5 m/s, including wind/J2 | Pass |
| WGS84 conversions | Round trip height within 1e-5 m, including poles and 500 km | Pass |
| J2 gravity | Equator/pole analytical values and independent off-axis potential gradient within 2e-8 m/s^2 | Pass |
| One unforced spherical orbit | Relative energy drift < 2e-9, closure < 0.2 m | 2.00e-15 and 3.03e-8 m |
| 6DoF rigid-body coast | Angular momentum and energy relative drift <= 1e-8 | Pass |
| Quaternion/torque checks | Analytic principal-axis spin, applied torque/I, thrust-offset moments, ECI/ENU attitude equivalence | Pass |
| Numerical convergence | Tightened RK45 tolerance improves vacuum delta-V and meets absolute criterion | Pass |

Additional tests cover transonic continuity, signed-alpha parity, Mach-dependent lift, hypersonic limits, all three wind components, zero-speed forces, propulsion curve integrals, scheduled cutoff with residual fuel, curve depletion, unsupported launches, simultaneous/zero-time transitions, three-stage configurations, variable-inertia assumptions, invalid inputs, and preset mass/thrust plausibility.

## Profiling and vectorization

`tests/benchmark_physics.py` reproduces the measurements and emits JSON. The committed [measurement record](phase1_benchmark.json) includes versions, timings, function counts, and cumulative profiling output.

- A full nominal sounding ascent/descent took about **1.44 s under cProfile**, using 3548 RHS evaluations and 264 stored adaptive samples.
- 10000 atmosphere evaluations took about **0.00082 s in one batch**, versus **0.384 s in a Python scalar loop**, approximately **467x** in that measurement. This is a machine-specific observation, not a CI speed threshold.
- Atmosphere layer selection/evaluation, coefficient curves, dynamic pressure, geodetic batches, and orbital elements use NumPy broadcasting/vectorized expressions. ENU-to-ECI history transforms are vectorized over samples.
- Profiling identified scalar force evaluation as the principal cost. An immutable default control input is now reused instead of reconstructed/revalidated for every RHS call. The sequential adaptive solver and event state machine were deliberately left sequential; batching them would change their semantics. Stage-mode-specific telemetry uses the same verified force evaluator rather than a second physics implementation.

## Nominal scenario and limitations

For the documented complete-stage Black Brant IX-class **estimate**, default vertical 3DoF ascent gives sampled apogee about **215.8 km** and impact near **454.61 s**. Event epochs are MECO/separation at 6 s, SES-1 at 12 s, and SECO at 39.5 s. This is a model result, not an independently validated flight benchmark. The exact apogee is not separately root-located; the reported maximum uses adaptive samples.

The first motor-only dry-mass estimate yielded about 317 km. The preset now explicitly budgets assumed non-motor vehicle/recovery hardware, and the evidence does not treat this change as empirical validation. Manufacturer-class presets remain approximate. In particular, Electron/Falcon default vertical runs do not demonstrate orbital insertion; they require a future guidance program.

The [model notes](physics_models.md) describe the upper-atmosphere surrogate, empirical aerodynamic assumptions, rotating-frame conventions, idealized changing-inertia/zero-efflux-momentum model, fixed force offsets, terrain/epoch limitations, and undefined orbital diagnostics. OpenRocket and published-flight comparisons remain unchecked in Phase 4. No claim of research-grade empirical fidelity is made solely from passing analytical tests.

One test collection attempt encountered a Windows Application Control rejection while loading a SciPy interpolation DLL. The suite was re-run successfully with approved execution outside the sandbox, and subsequent ordinary runs also passed. There is no remaining test blocker.
