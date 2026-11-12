# Aerospace Trajectory, Guidance & Mission Analysis Suite

Architecture is initialized; all development phases below remain unimplemented. This roadmap is a plan, not evidence of research readiness. Execute authorized phases in order and satisfy the `AGENTS.md` audit gate before proceeding.

## Phase 1: High-Fidelity 3DoF/6DoF Physics & Environment

- [ ] Define typed SI state/configuration interfaces, coordinate frames, time conventions, and 3DoF/6DoF model boundaries.
- [ ] Implement US Standard Atmosphere 1976 and a documented upper-atmosphere extension with validity ranges, source tables, and continuous transitions where physically appropriate.
- [ ] Implement WGS84 geometry and Earth constants, J2 gravity, and consistent rotating-frame Coriolis/centrifugal terms; verify against an inertial-frame formulation.
- [ ] Implement mass depletion, multistage properties, nozzle ambient-pressure compensation, and Mach/alpha aerodynamic drag curves with provenance and interpolation limits.
- [ ] Implement translational and rigid-body rotational dynamics with documented force, moment, inertia, and center-of-mass conventions.
- [ ] Build `solve_ivp` propagation with terminal events for engine cutoff, stage separation, and ground impact; apply transitions and restart without timestep-triggered approximations.
- [ ] Audit: pytest coverage, Tsiolkovsky delta-V, US76 reference tables, gravity/frame checks, event accuracy, and tolerance convergence; profile/vectorize appropriate bottlenecks, document evidence, and commit a clean milestone.

## Phase 2: Guidance, Navigation & Control

- [ ] Implement launch-rail constraints and release, pitch-kick, and zero-lift gravity-turn modes with explicit transition conditions.
- [ ] Implement Powered Explicit Guidance and Linear Tangent Steering with documented assumptions, target constraints, and feasibility handling.
- [ ] Define navigation truth/estimate interfaces, sensor assumptions, and update rates.
- [ ] Implement quaternion attitude kinematics with declared ordering, rotation direction, normalization policy, and singularity-free attitude errors.
- [ ] Implement PID and LQR thrust-vector-control autopilots with actuator limits, rate limits, anti-windup where applicable, and stability evidence.
- [ ] Couple guidance/control to propagation with documented sample-and-hold timing and event consistency.
- [ ] Audit: pytest guidance transitions, quaternion benchmarks, equilibrium/step responses, and closed-loop trajectory checks; re-run applicable Phase 1 benchmarks, profile/vectorize, document limitations, and commit a clean milestone.

## Phase 3: Parallelized Monte Carlo, Sensitivity & Optimization

- [ ] Define uncertainty distributions, correlations, output metrics, and deterministic per-run random seeds.
- [ ] Implement `ProcessPoolExecutor` dispersion runs with Windows-safe entry points, reproducible worker behavior, and explicit failed-run reporting.
- [ ] Plot 1/2/3-sigma covariance ellipses with declared projection, scaling, and sample size; distinguish sigma radii from joint confidence probabilities.
- [ ] Implement Sobol sensitivity analysis with sampling design, convergence checks, and uncertainty estimates; document any additional dependency before introducing it.
- [ ] Implement `scipy.optimize` staging and pitch optimization with bounds, constraints, scaling, feasibility checks, and reproducible baselines.
- [ ] Audit: pytest serial/parallel reproducibility, synthetic covariance and known sensitivity benchmarks, optimizer feasibility and improvement; profile/vectorize bottlenecks, record evidence, and commit a clean milestone.

## Phase 4: Verification & Validation Suite & PDF Technical Report

- [ ] Consolidate Tsiolkovsky, orbital-energy conservation under applicable conservative assumptions, and event/convergence regression tests.
- [ ] Compare compatible cases against OpenRocket and published Falcon 9 benchmarks with cited sources, matched assumptions, uncertainty bounds, and clearly identified unavailable inputs.
- [ ] Separate analytical verification, cross-tool comparison, and observational validation; define acceptance criteria and explain discrepancies without tuning to unsupported targets.
- [ ] Generate `Engineering_VV_Report.pdf` automatically with ReportLab, including equations/assumptions, inputs and units, plots, pass/fail tables, tolerances, references, seeds, dependency versions, and Git revision.
- [ ] Render and inspect the generated PDF for layout, readable figures, complete tables, and traceable conclusions.
- [ ] Audit: pytest report data and numerical regressions, all applicable analytical benchmarks, report reproducibility, and measured performance; document unresolved limitations and commit a clean milestone.

## Phase 5: Mission Control Streamlit App & Playwright Demo Automation

- [ ] Build Streamlit mission configuration, trajectory visualization, guidance diagnostics, dispersion, optimization, and V&V report views using shared public APIs.
- [ ] Show explicit units, model validity, progress, cancellation/error states, and reproducible exportable run configurations; use Plotly for interactive results.
- [ ] Add Playwright demo automation for a documented mission workflow, deterministic inputs, screenshots, and report download verification. Add Playwright and its browser installation instructions when this phase begins.
- [ ] Document environment setup, launch commands, demo execution, and scientific limitations.
- [ ] Audit: pytest application-facing logic, Playwright workflow checks, end-to-end benchmark consistency, and responsive performance; profile/vectorize relevant bottlenecks, document evidence, and commit a clean milestone.

## Initial environment

Create a local `.venv`, then install dependencies with `python -m pip install -r requirements.txt` using that environment's interpreter. The initial requirements are intentionally unpinned as requested; record or lock validated versions when numerical implementation begins. Browser tooling is introduced in Phase 5. Keep local environments, credentials, and generated caches out of Git.
