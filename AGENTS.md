# Repository engineering rules

These rules apply to all work in this repository. Phases 1, 2, and 3 were explicitly authorized and are implemented; see `docs/phase1_audit.md`, `docs/phase2_audit.md`, and `docs/phase3_audit.md`. Do not advance into a later roadmap phase without authorization. Preserve the documented distinction between analytical verification and empirical validation.

## Architecture

- `src/physics/`: vehicle dynamics, propulsion, aerodynamics, environment, reference frames, and event-driven propagation.
- `src/guidance/`: trajectory guidance, navigation interfaces, attitude control, and actuators.
- `src/analysis/`: Monte Carlo, sensitivity, optimization, verification metrics, and reporting.
- `src/ui/`: Streamlit presentation and visualization; consume public model APIs rather than duplicating physics.
- `tests/`: isolated unit tests, analytical benchmarks, regression tests, and integration tests.
- `docs/`: model assumptions, equations, references, validation evidence, and usage documentation.
- `assets/`: documented reference data and presentation resources with provenance and licensing.

Keep physics independent of guidance, analysis, and UI. Use explicit interfaces and configuration objects; avoid hidden mutable state and circular imports.

## Scientific Python standards

- Require type hints on every function and method, including return types, and on model/configuration fields. Type NumPy arrays explicitly where practical and document shape and dtype.
- Require NumPy-style docstrings on modules, classes, and functions, including Parameters, Returns, Raises, and Notes where applicable.
- Specify explicit SI units for every variable, parameter, state component, field, and output. Use unit-bearing names such as `altitude_m`, `velocity_m_s`, and `mass_kg`; also document units in docstrings and state schemas. Mark dimensionless quantities, indices, and flags explicitly as dimensionless or not applicable. Angles are radians internally.
- Convert non-SI input at system boundaries only. Record frame, origin, axis convention, epoch/time basis, quaternion convention, and array ordering alongside units.
- State assumptions, validity domains, numerical tolerances, and source provenance. Never fabricate validation data or present unvalidated results as validated.

## Integration and discontinuities

- Use `scipy.integrate.solve_ivp` with root-located terminal event functions for stage separation, engine cutoff, and ground impact. Set `terminal = True` and appropriate crossing directions; document event units and handling of initial roots and coincident events.
- Stop at each discontinuity, apply the discrete state/configuration change once, and restart integration for the next segment. Ground impact terminates the mission propagation.
- Do not use timestep-based `if/else` checks, state clipping, or arbitrary step counts as substitutes for physical event detection. Ordinary branching for dispatch and discrete state updates is allowed.
- Demonstrate tolerance convergence and event-time accuracy. Keep inertial and rotating-frame formulations consistent; do not double-count centrifugal or Coriolis accelerations.

## Mandatory self-audit after every implementation phase

1. Write and run `pytest` unit tests covering new behavior, edge cases, failure modes, and event transitions. Keep randomness seeded and tests reproducible.
2. Verify applicable analytical or authoritative benchmarks, including Tsiolkovsky delta-V and US Standard Atmosphere 1976 tables. Add conservation and frame-consistency checks where appropriate. Declare acceptance thresholds before judging results.
3. Profile representative workloads, vectorize slow loops where numerically appropriate, and re-run accuracy tests. Do not force-vectorize sequential integrator/control logic or optimize without evidence.
4. Record commands, results, deviations, model limitations, and performance evidence in `docs/`. Resolve audit failures before marking the phase complete.
5. Update `ROADMAP.md` checkboxes only with supporting evidence. Review the diff, exclude secrets/generated environments, and commit a clean milestone to Git. Do not push unless authorized.

The architecture-only initialization requires structure and dependency checks; analytical tests begin when the corresponding models exist. Never add placeholder passing tests or claim unimplemented models are verified.

## Autonomous execution

Once implementation is authorized, proceed through the requested scope in dependency order with the audit gate after each phase. Preserve unrelated user work. Ask for missing scientific inputs that materially affect validity; document justified provisional assumptions. Keep numerical kernels out of UI code and make analysis runs reproducible from stored configuration, seeds, dependency versions, and Git revision.
