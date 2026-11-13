# Phase 3 implementation audit

Audit date: 2026-11-12. This milestone implements dispersion, sensitivity, and
constrained conceptual sizing. See [analysis models](analysis_models.md) for
statistical conventions, SI APIs, sampling costs, and physical limitations.
The associated revision is the commit containing this audit and its artifacts;
recorded benchmark metadata identifies the base revision plus a dirty tree.

## Verification commands and acceptance

Run with the repository `.venv` interpreter:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m tests.benchmark_analysis
git diff --check
```

The full suite passed **145 tests in 115.57 s**, retaining all 125 prior tests,
including US76 tables, Tsiolkovsky, mass conservation, rotating-frame equivalence,
guidance insertion, and solver convergence. New assertions specify thresholds
before accepting results. A final design-bounds guard was also checked with the
optimizer acceptance suite. Compilation, type-hint/docstring AST checks,
dependency checks, and diff checks passed. No new dependency was needed.

| Check | Acceptance/evidence |
| --- | --- |
| 23-input default marginals | 10000 seeded draws: means within 0.04 sigma, standard deviations within 3.5%; all uniform samples in bounds |
| Child streams | Exact prefix agreement across batch sizes; no global RNG |
| Correlated normals | Requested rho=0.7 within 0.02; singular matrices rejected |
| Physical perturbations | Independent thrust/Isp ratios, constant-flow and curve modes; full Cd scaling and unchanged Cl; nominal configuration immutable |
| Wind shear | Superposition matches analytical piecewise-linear values, including nominal knots |
| Covariance contours | Analytical rotated covariance; every vertex's squared Mahalanobis radius within 1e-12; correct joint Gaussian coverage; degenerate rank retained |
| Impact map | Known equatorial arcs and dateline crossing within 1e-7 m in the declared spherical projection |
| Parallel execution | Spawned 2-worker and serial metrics agree exactly on deterministic fixtures; invalid realization remains at its original index |
| Censoring/export/plot | Horizon runs do not become impacts/apogees; missing counts and strict JSON nulls preserved; headless PNG generation |
| Numerical metrics | Halved-step refined apogee within 0.02 m and Q within 1% on fixture; exact pre-cutoff G-load included |
| Ground root robustness | Fast drop with coarse trial stages matches fine impact epoch within 1e-5 s, surface residual within 1e-6 m |
| Regression | Orthogonal linear model gives analytical 0.2/0.8 contributions; correlated-input attribution withheld; singular/constant cases handled |
| Sobol | Ishigami first/total indices within 0.025 of analytical values; additive model within 0.005; half-sample estimate within 0.01; seeded intervals reproduce exactly |
| Physical Sobol adapter | Complete 24-launch structured design for three metrics; incomplete trajectories rejected without dropping rows |
| Sizing | Fuel conserved, engine flow unchanged, explicit dry scaling; SLSQP and DE improve real simulated fixture apogee while satisfying verified constraints |
| Payload | Two-stage airborne fixture increases payload and independently achieves final-stage insertion; early horizon cannot qualify |
| Failure/bounds/budget | Infeasible constraints, missing apogee, absent orbital target, and out-of-bounds design cannot be labeled feasible; search evaluation cap respected |

## Workload and performance evidence

[Benchmark JSON](phase3_benchmark.json), [100-run ensemble](phase3_ensemble.json),
[optimization record](phase3_optimization.json), and
[dispersion plot](phase3_dispersion.png) retain inputs/results. The plot was
rendered and visually inspected. Versions: Python 3.12.14, NumPy 2.5.4,
SciPy 1.18.1 on Windows; the record includes root seeds and Git provenance.

- **100 Black Brant-class dispersed launches:** 100 impacts, zero failed or
  horizon-only runs; four processes; 39.75 s. Mean apogee 215330.5 m, standard
  deviation 2735.9 m. This is the model's response to assumed tolerances, not
  an observed flight distribution.
- **1000 short test launches:** 1000 impacts, zero failures; four processes;
  64.99 s. This exercises batch scale and IPC using the explicitly nonrealistic
  low-Isp test fixture; it is not a claim of 1000 full-size mission timings.
- **Profiled nominal sounding trajectory:** 1.48 s, 2000 RHS evaluations;
  force evaluation consumed 1.38 s cumulatively. Guidance navigation and scalar
  atmosphere/force calls dominate; process overhead is relevant for short runs.
- Marginal transforms, correlation mapping, covariance/eigendecomposition,
  contour vertices, metric reductions, standardized regression, and sensitivity
  matrix construction use array operations. Independent launches parallelize;
  sequential event integration and controller memory remain sequential.
  Histories are reduced inside workers to avoid large IPC/memory costs. These
  measurements do not establish a universal core-scaling factor.

## Optimization evidence and interpretation

The nominal sounding case has 1768 kg total propellant and 650 kg payload.
The constrained SLSQP benchmark allows S1 fuel fraction 0.3–0.65 and pitch
0–8 degrees. Dry/fuel ratios are held fixed. Example load limits are set to
1.05 times nominal full-flight sampled loads: approximately **2.089 MPa Q** and
**11.808 g**. These deliberately generous numerical-study limits are not
manufacturer structural allowables. Q includes ballistic descent to impact;
the current fixed-geometry surrogate predicts much higher descent Q than
typical launch-ascent limits.

The verified candidate moved S1 fuel fraction from 0.429864 to 0.494638 and
pitch-kick from 0.0872665 to 0.0863501 rad. Apogee improved from **215508.75 m
to 222248.27 m** (about 3.13%). Verified maxima were 2.02873 MPa and 11.7535 g,
within the chosen limits. Search used 18 unique designs in about 20.22 s,
then repeated candidate and baseline at half max-step and tighter rtol.
It reached the configured five-iteration limit: **feasible and improved,
but optimizer_converged=False**. No global optimum is claimed.

Independent-input regression screening on the 100-launch ensemble gives R^2
of 0.99990 (apogee), 0.99993 (full-flight Max-Q), and 0.99979 (downrange).
The strongest absolute coefficients are stage-2 Isp for apogee, Cd multiplier
for Max-Q, and elevation error for downrange. These are local linear findings
for this scenario. Finite-sample input correlations mean squared coefficients
need not sum to one; they are not reported as exact Sobol variance shares.
Actual Sobol implementation is separately verified on analytical additive and
Ishigami interaction benchmarks. A full 23-input flight Sobol convergence study
has not been performed; its required sampling cost is documented.

## Issues resolved and remaining limits

The tests exposed unseeded confidence-interval resampling in SciPy's convenience
bootstrap API. Explicit paired, seeded resampling fixed it. Headless plotting
now uses an Agg canvas instead of relying on local Tk installation. Symmetric
elevation dispersions now pass through vertical without reversing the intended
pitch program. Cd uncertainty includes incidence drag.

Coarse descent integration exposed trial evaluations below the atmosphere
domain before ground root location. A constant environment extension for those
trial states preserves root finding without clipping position/velocity or
substituting a step-based impact condition. Exact left-side event loads now
prevent burnout peaks disappearing from a right-continuous history.

Uncertainty widths, correlation defaults, aerodynamic/inertia/geometry inputs,
and sizing relationships are assumptions. Covariance contours are conditional
on observed impacts, with finite-sample covariance uncertainty; they are not
certified safety or tolerance regions. Apogee interpolation and sampled loads
need resolution studies for new missions. A feasible optimum is nominal, not
dispersion-robust. Reporting/PDF generation and the UI remain future phases.
