# Phase 2 implementation audit

Audit date: 2026-11-12. Phase 2 is implemented within the assumptions in
[guidance models](guidance_models.md). This is numerical verification, not
flight-data validation. The milestone is the commit containing this report.

## Verification

`python -m pytest -q`: **111 passed in 67.70 s**, including all 68 Phase 1
cases. `python -m compileall -q src tests`, `python -m pip check`, an AST
argument/return-annotation and docstring audit, and `git diff --check` passed.
Use `.venv/Scripts/python.exe` on Windows. No dependencies were added.

Acceptance criteria are asserted in the tests, rather than selected after
examining results:

| Check | Acceptance |
| --- | --- |
| Inclined rail, ECI/ENU and 3DoF/6DoF | Release at 15 m within 1e-6 m; lateral displacement <1e-5 m; continuous state |
| Ideal air-relative gravity turn | Incidence <1e-7 rad with three-axis wind |
| Accepted control clocks | Independent of adaptive solver step size |
| PID/LQR shear tracking at 15 and 80 km initial altitude | Settled nose error <1 degree after 20 s; gimbals <=5 degrees per axis; slew <=20 degrees/s plus numerical tolerance |
| Quaternion | Analytical derivative/sign checks; norm error <=2e-8 in shear runs |
| Circular/elliptical upper-stage insertion, both DoF modes | Radius error <100 m; radial/tangential velocity errors <1 m/s |
| Full orbital launches | Electron/Falcon ideal and Electron 6DoF achieve insertion through staging/fairing/guided cutoff |
| Guided rotating frames and tolerances | ECI/ENU and rtol=1e-6 versus 1e-9 agree within 2 mm and 0.2 mm/s |
| Failure handling | Unreachable target reports failure; no false guided cutoff; invalid rail/nozzle configurations rejected |

Additional checks cover vis-viva, exact thrust-curve fuel integrals, analytic
linear-tangent pitch rates, controller moment signs and local stable poles,
anti-windup, loss of thrust authority, per-run memory isolation, and deterministic
reuse. Existing US76, Tsiolkovsky, mass conservation, event and convergence
benchmarks continue to pass.

## Reproducible measurements

`python -m tests.benchmark_guidance` emits the committed
[measurement record](phase2_benchmark.json), including versions, base Git
revision and working-tree status. The record belongs to this milestone's tree.
Python 3.12.14, NumPy 2.5.4 and SciPy 1.18.1 were used on Windows 11.

| 200 km target mission | Radius error [m] | Radial error [m/s] | Tangential error [m/s] |
| --- | ---: | ---: | ---: |
| Electron class, ideal | -0.000925 | -0.000903 | -4.58e-7 |
| Falcon class, ideal | -0.001018 | -0.000562 | -9.39e-7 |
| Electron class, PID 6DoF | -0.201723 | -0.051136 | 1.40e-6 |

For the 15 km shear case, maximum settled errors were 0.0001683 rad (PID)
and 0.00002538 rad (LQR), with slew at the configured 0.349066 rad/s limit.
The full Electron 6DoF run reached cutoff at 410.98645 s, with 73 feasible
plans. Its maximum gimbal was 0.06449 rad. Full-mission handoff transients
reached 0.3942 rad nose error; the settled shear criterion does not imply
uniform one-degree tracking through all phase changes.

## Corrections and performance

The audit found that rapid full pitch replanning near cutoff could demand
untrackable turns and excite command-rate derivative kicks. Filtered rate
feed-forward and a final five-second fixed-pitch, speed-only cutoff trim
resolved the full 6DoF insertion failure. This full launch is now a regression
test. Roundoff-separated event timestamps also produced misleading numerical
slew estimates; output coalescing preserves right-hand states while retaining
exact roots in the event ledger. Acceptance limits were not relaxed.

The profiled 25 s PID shear run took 4.10 s; force evaluation accounted for
3.12 s cumulative. Ideal launch runs took 3.60/3.79 s and Electron 6DoF took
32.41 s locally. Timings are observations, not CI speed limits. The hot force
path reuses control data without reconstructing validated inputs. Idle final
coast clocks are skipped, and noninitial guided segments use a 0.1 s initial
trial step. NumPy vectorizes orbital histories, frame transforms, fuel-curve
integration, and telemetry reductions. Adaptive propagation and sampled
controller memory remain sequential to preserve event semantics.

## Limits

The planner is numerical linear-tangent shooting inspired by PEG, not flight
PEG. It assumes central-gravity vacuum prediction and uses truth navigation.
Preset inertia/nozzle geometry is provisional; two-axis TVC supplies no roll
or unpowered attitude authority. LQR stability evidence is local. No empirical
flight validation, sensor estimator, mission optimization, or UI is claimed.
See the model documentation for target/frame definitions and feasibility
fallback behavior. Phases 3–5 remain unimplemented.
