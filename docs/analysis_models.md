# Phase 3: uncertainty, sensitivity, and constrained sizing

The analysis package calls the shared physics and guidance APIs. It does not
duplicate force models or alter event timing. This phase supplies engineering
analysis tools and numerical verification; it does not establish calibrated
vehicle uncertainty or an operational range-safety envelope.

## Monte Carlo launches and replay

Save this example as a script and run it from the repository root. The main
guard is required for Windows process spawning. `workers=1` is the serial path.

```python
from src.analysis import run_monte_carlo, plot_dispersion, regression_from_ensemble
from src.guidance import GuidanceProgram
from src.physics import PRESETS, SimulationConfig

if __name__ == "__main__":
    vehicle = PRESETS["Two-Stage Sounding Rocket (Black Brant IX Class)"]
    simulation = SimulationConfig(duration_s=650., max_step_s=4., rtol=1e-7)
    ensemble = run_monte_carlo(vehicle, simulation, GuidanceProgram(),
                               count=100, seed=20261112, workers=4)
    ensemble.save("docs/ensemble.json")
    plot_dispersion(ensemble, "docs/impact_ellipses.png")
    screening = regression_from_ensemble(ensemble)
    print(ensemble.metadata["status_counts"], screening.r_squared)
```

The default two-stage design has **23 independent inputs**:

| Inputs | Distribution | Meaning of the specified range |
| --- | --- | --- |
| Each stage's vacuum thrust | Gaussian | +/-3% is three sigma |
| Each stage's vacuum Isp | Gaussian | +/-1.5% is three sigma |
| Each stage's propellant mass | Gaussian | +/-1.5% is three sigma |
| Each stage's dry mass | Gaussian | +/-2% is three sigma |
| Launch azimuth and elevation | Uniform | +/-0.5 degrees, converted to radians |
| Complete drag coefficient multiplier | Uniform | +/-10%, lift unchanged |
| East/north/up gust at 0/10/30/80 km | Gaussian | Each component +/-15 m/s is three sigma |

These ranges are requested provisional assumptions, not distributions inferred
from manufacturer data. Gaussian tails are unbounded. Physically invalid draws
are retained as failed runs; they are not clipped or resampled. Use
`DispersionSpace` and named `Uncertainty` objects to select marginals/ranges or
provide a positive-definite latent Gaussian correlation matrix. This is a
Gaussian copula: uniform Pearson correlations differ from latent correlations.
Wind knots/components are independent by default; correlations can encode a
better supported spatial/coherent gust model. The model represents frozen
altitude shear, not a time-varying turbulence spectrum.

Thrust and Isp are independent: constant mass flow scales by
`thrust_multiplier / isp_multiplier`. Thrust curves scale in force at unchanged
times. Mass perturbations change masses, retaining nominal geometry and explicit
cutoff schedules; a thrust curve can therefore leave residual fuel. Cd scales
the entire curve, including incidence drag, while Cl is unchanged. Gust and
nominal wind are interpolated on the union of both sets of knots. Rail elevation
may pass through vertical while keeping the nominal pitch-program azimuth;
this avoids turning a positive elevation error into an opposite pitch program.

`SeedSequence` creates a separate seed per run. Inputs are sampled before
dispatch; results are returned in input order. Changing worker count or batch
length preserves the existing sample prefix. Only compact metrics return from
workers, so 1000 runs do not retain 1000 trajectory histories. Default worker
count is min(CPU count,8). Set an explicit count to manage machine load and
external BLAS thread settings. Parallel numerical/configuration exceptions are
recorded with their run index and message.

JSON includes nominal vehicle/simulation/guidance, uncertainty definitions and
units, every input row, every replay seed, outcomes, dependency versions, Git
revision and dirty flag. `evaluate_design` reruns a stored input matrix with
the nominal configuration and `seeds=tuple(run.seed for run in result.runs)`.
An individual row can be applied using `perturb_launch`, then `simulate`.
Saved nonfinite/missing metrics become JSON null, never nonstandard NaN tokens.

## Metrics and covariance contours

Each run is `impact`, `horizon`, or `failed`. A successful integration ending at
the requested time is not assumed to have impacted. Statistics report a finite
sample count and missing count for each metric. Their means, sample standard
deviations (ddof=1), and empirical 2.5/50/97.5 percentiles are conditional on that
metric being observed. Percentiles are output-distribution summaries, not
confidence intervals for an estimated mean. Insertion success is counted
separately from integration success.

Downrange/crossrange use the terminal ground-impact state transformed to
Earth-fixed coordinates at impact time. WGS84 geodetic latitude/longitude are
mapped to a **spherical azimuthal-equidistant projection** with WGS84 equatorial
radius. Arc length and initial bearing give distances along the fixed nominal
launch azimuth and positive to its right. Each run uses the same range axis,
not its perturbed launch azimuth. This is not an exact ellipsoidal geodesic;
the launch antipode is undefined and rejected. Dateline crossings are supported.

For n >=2 observed impacts, covariance is the unbiased 2 by 2 sample covariance
in m^2. With eigendecomposition C=V diag(lambda) V^T, a k-sigma contour is
`center + V diag(k*sqrt(lambda)) [cos(theta),sin(theta)]`. Returned semiaxes
are ordered major/minor, orientation is radians counterclockwise from downrange,
and contour coordinates are in metres. Geometry is exact for the estimated
covariance; population covariance remains uncertain at finite n.

For a nonsingular bivariate Gaussian, probability inside Mahalanobis radius k
is `1-exp(-k^2/2)`: 39.35%, 86.47%, 98.89% at k=1,2,3. These are not the
univariate 68/95/99.7% intervals. Degenerate covariance is returned as a line or
point with its rank and undefined bivariate coverage. No impacts or one impact
produce no ellipse. Multimodal/non-Gaussian clouds and mixed failure/impact
outcomes need analysis beyond a fitted covariance contour. Plotting uses a
headless Matplotlib canvas and includes the outcome counts.

Apogee uses a cubic Hermite height interpolant with geodetic vertical velocity
to refine interior maxima. It is an interpolation estimate, not a new physical
terminal event. If no maximum is traversed before the horizon, apogee is missing;
the highest altitude within the simulated horizon is reported separately.
Max-Q and maximum absolute axial G use adaptive samples **and left-side event
loads**, so the pre-cutoff thrust peak survives right-continuous output stitching.
Unobserved between-sample peaks still require step convergence. Actual orbital
errors are recomputed at final-stage cutoff against spherical insertion radius,
radial velocity zero and vis-viva speed. Missing cutoff does not imply success.

## Sensitivity: screening and Sobol analysis

`standardized_regression` fits centered, standardized least squares. It returns
signed standardized coefficients, R^2, empirical input correlations, and a
condition number. With explicitly independent inputs, coefficient squares are
reported as **linear variance approximations**. They are not renamed Sobol
indices, normalized to sum to one, or offered for correlated designs. Low R^2
indicates missed nonlinearity/interactions. Constant outputs have undefined
indices; constant/collinear inputs and missing data are rejected.

`sobol_analysis` computes genuine first/total-order indices with a scrambled
Sobol pick-freeze design and SciPy's Saltelli estimator. Inputs must be
independent. `n_base` is a power of two and requires `n_base*(d+2)` model runs.
The default 23 inputs and n_base=256 require 6400 launches; use regression to
screen or select a scientifically justified smaller subspace before a large run.
The callable accepts an (n,d) SI input matrix and returns an (n,s) metric matrix.
`launch_sobol` connects this API directly to the process-pool launch evaluator:

```python
from src.analysis import DispersionSpace, Uncertainty, launch_sobol

# Put this call inside the same main guard as the Monte Carlo example.
space = DispersionSpace((Uncertainty("stage_1_isp_fraction", .015),
                         Uncertainty("stage_2_isp_fraction", .015)))
# indices = launch_sobol(vehicle, simulation, GuidanceProgram(), space,
#                        n_base=64, seed=12, workers=4)
```

Indices, raw SI design/evaluations, half-sample estimates, and seeded paired
bootstrap percentile intervals are retained. Intervals are Monte Carlo
uncertainty diagnostics, not a rigorous coverage guarantee for a single QMC
scramble. Check increasing n and independent seeds for a substantive study.
Raw negative or >1 estimates are retained to reveal estimator noise. The
Ishigami analytical benchmark verifies interactions. Constant outputs remain
NaN; no failed or missing row is silently dropped from a structured design.

## Constrained trajectory and vehicle sizing

`optimize_trajectory(vehicle, simulation, guidance, OptimizationConfig(...))`
supports SLSQP and seeded differential evolution. Decision variables are:

- S1 propellant / (S1+S2 propellant), strictly between 0 and 1;
- pitch-kick angle [rad];
- payload [kg], when maximizing payload-to-orbit.

Exactly two stages are required. Total propellant is fixed. By default each
stage keeps its baseline dry/fuel ratio; `scale_dry_mass=False` fixes dry masses.
Engine flow/Isp/nozzle/aerodynamic geometry remain fixed. Thrust-curve times and
explicit cutoff times stretch with allocated fuel at fixed thrust. This is a
conceptual allocation model with provisional geometry, not detailed tank sizing.

Max-Q [Pa] and maximum absolute axial load [g0] are explicit nonlinear constraints
over the **entire simulated mission**, including descent if propagated to impact.
They are not restricted to powered ascent. Default ceilings are example inputs;
they can be infeasible for a supplied mission. A suborbital reentry can have a
much higher Q than ascent in the current fixed-geometry aerodynamic surrogate.
Suborbital apogee must be observed before the horizon. Payload-to-orbit also
requires the target radius and both velocity errors to meet `OrbitTarget`
tolerances at final-stage cutoff. Supply an orbital `GuidanceProgram` for payload
and a target-free program for apogee. A larger payload without actual insertion
is infeasible.

```python
from src.analysis import OptimizationConfig, optimize_trajectory, write_json

settings = OptimizationConfig(method="SLSQP", objective="apogee",
                               max_q_pa=100_000., max_axial_g=20.,
                               max_iterations=10, max_evaluations=60)
# sizing = optimize_trajectory(vehicle, simulation, GuidanceProgram(), settings)
# write_json("docs/sizing.json", sizing)
```

Variables are normalized for optimization; SLSQP uses a 1e-3 normalized finite
difference step to avoid differentiating integrator roundoff. Exact repeated
design evaluations are cached between objective and constraints. DE uses one
worker internally; ensemble parallelism is separate. Evaluation budgets count
unique search designs, excluding baseline and final verification runs.

The best search-feasible design is repropagated at half max-step and tighter
rtol, as is the baseline. `candidate.feasible`, `improved`, and
`optimizer_converged` are separate. A budget/iteration stop may yield a useful
feasible improvement while convergence remains false. A failed final check is
reported as infeasible. No global optimum or robustness to dispersions is
claimed; optimized designs should subsequently be dispersed and checked at
additional resolution. Objective/structural limits use the documented metric
approximations, not certified continuous-time peak bounds.

## References and evidence

- [SciPy Sobol indices](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sobol_indices.html): independent-input Saltelli estimator, sampling cost, and Ishigami example.
- [SciPy constrained optimization](https://docs.scipy.org/doc/scipy/tutorial/optimize.html) and [differential evolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html): supported algorithms and constraints.
- [NIST bivariate normal regions](https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/bvntolpl.htm): distinction between covariance geometry and finite-sample tolerance regions.
- [Penn State multivariate normal notes](https://online.stat.psu.edu/stat505/Lesson04): covariance eigenvectors and chi-square probability contours.

See [Phase 3 audit](phase3_audit.md) for acceptance thresholds, actual workload
counts, profiling, achieved optimization, and remaining limitations.
