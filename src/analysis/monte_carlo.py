"""Seeded parallel launch dispersion, conditional statistics, and covariance ellipses.

Notes
-----
Default uncertainty ranges are provisional engineering assumptions, not measured
vehicle tolerances. Gaussian half-ranges mean three standard deviations;
uniform half-ranges are hard bounds. Windows callers must use a __main__ guard.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import multiprocessing
import os
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr, ndtri

from src.guidance import GuidanceProgram
from src.physics import SimulationConfig, VehicleConfig, WindProfile, simulate
from .metrics import TrajectoryMetrics, provenance, trajectory_metrics, write_json

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Uncertainty:
    """Named zero-centered additive perturbation with declared units.

    Parameters
    ----------
    name : str
        Stage names use stage_1_thrust_fraction, isp_fraction,
        propellant_fraction, dry_fraction (and stage_2, etc.); global names
        are azimuth_rad, elevation_rad, cd_fraction, wind_{node}_{e/n/u}_m_s.
    half_range : float
        Three-sigma width for Gaussian or bound for uniform, in ``unit``.
    distribution : {"gaussian", "uniform"}
        Marginal distribution selector (dimensionless).
    unit : {"1", "rad", "m/s"}
        Fractions are dimensionless, angles radians, wind speed m/s.
    """

    name: str
    half_range: float
    distribution: Literal["gaussian", "uniform"] = "gaussian"
    unit: str = "1"

    def __post_init__(self) -> None:
        """Require finite positive uncertainty scales and supported marginal types."""
        if not self.name or not np.isfinite(self.half_range) or self.half_range <= 0.:
            raise ValueError("Uncertainty needs a name and positive finite half-range")
        if self.distribution not in ("gaussian", "uniform") or self.unit not in ("1", "rad", "m/s"):
            raise ValueError("Unsupported distribution or unit")


@dataclass(frozen=True)
class DispersionSpace:
    """Ordered uncertainties and optional Gaussian-copula correlation matrix.

    Attributes
    ----------
    parameters : tuple of Uncertainty
        Marginals; duplicate names are rejected.
    wind_altitude_m : tuple of float
        Gust interpolation knots [m], added to the nominal wind profile.
    correlation : tuple of tuples or None
        Positive-definite unit-diagonal latent-normal correlation, dimensionless.
        None means independent inputs. Uniform marginal Pearson correlations
        differ from latent correlations; classical Sobol requires independence.
    """

    parameters: tuple[Uncertainty, ...]
    wind_altitude_m: tuple[float, ...] = (0., 10000., 30000., 80000.)
    correlation: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        """Validate names, wind heights [m], and a nonsingular correlation model."""
        if not self.parameters or len({p.name for p in self.parameters}) != len(self.parameters):
            raise ValueError("Specify nonempty, uniquely named uncertainties")
        heights_m = np.asarray(self.wind_altitude_m)
        if heights_m.ndim != 1 or len(heights_m) < 1 or not np.all(np.isfinite(heights_m)) or np.any(np.diff(heights_m) <= 0.):
            raise ValueError("Wind knots must be finite and increasing")
        if self.correlation is not None:
            matrix = np.asarray(self.correlation)
            if (matrix.shape != (len(self.parameters),)*2 or not np.all(np.isfinite(matrix))
                    or not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.)
                    or not np.allclose(np.diag(matrix), 1., atol=1e-12, rtol=0.)):
                raise ValueError("Correlation must be symmetric with unit diagonal")
            try:
                np.linalg.cholesky(matrix)
            except np.linalg.LinAlgError as error:
                raise ValueError("Correlation must be positive definite") from error

    @property
    def names(self) -> tuple[str, ...]:
        """Return ordered parameter identifiers (dimensionless metadata)."""
        return tuple(p.name for p in self.parameters)

    def transform(self, unit_samples: ArrayLike) -> FloatArray:
        """Map open-unit-cube samples (n,d) to physical perturbations in each column's units."""
        unit = np.asarray(unit_samples, dtype=float)
        if unit.ndim != 2 or unit.shape[1] != len(self.parameters) or not np.all(np.isfinite(unit)) or np.any((unit <= 0.) | (unit >= 1.)):
            raise ValueError("Samples must have shape (n,d) strictly within (0,1)")
        normal = ndtri(unit)
        if self.correlation is not None:
            normal = normal @ np.linalg.cholesky(self.correlation).T
        values = np.empty_like(normal)
        for index, parameter in enumerate(self.parameters):
            values[:, index] = (normal[:, index]*parameter.half_range/3. if parameter.distribution == "gaussian"
                                else (2.*ndtr(normal[:, index])-1.)*parameter.half_range)
        return values


def default_dispersion_space(stage_count: int = 2) -> DispersionSpace:
    """Return 4 uncertainties per stage plus angles, drag, and 12 ENU wind-knot values.

    Notes
    -----
    Thrust +/-3%, Isp/fuel +/-1.5%, dry +/-2% are Gaussian three-sigma ranges.
    Azimuth/elevation +/-0.5 degree and complete Cd +/-10% are uniform. Wind
    knots have independent Gaussian components with +/-15 m/s three-sigma range.
    """
    if not isinstance(stage_count, int) or stage_count < 1:
        raise ValueError("Stage count must be positive")
    parameters = [Uncertainty(f"stage_{stage}_{name}_fraction", width)
                  for stage in range(1, stage_count+1)
                  for name, width in (("thrust", .03), ("isp", .015), ("propellant", .015), ("dry", .02))]
    parameters.extend((Uncertainty("azimuth_rad", np.deg2rad(.5), "uniform", "rad"),
                       Uncertainty("elevation_rad", np.deg2rad(.5), "uniform", "rad"),
                       Uncertainty("cd_fraction", .1, "uniform")))
    parameters.extend(Uncertainty(f"wind_{node}_{axis}_m_s", 15., "gaussian", "m/s")
                      for node in range(4) for axis in ("e", "n", "u"))
    return DispersionSpace(tuple(parameters))


def sample_inputs(space: DispersionSpace, count: int, seed: int) -> tuple[FloatArray, tuple[int, ...]]:
    """Return perturbations (count,d) and independent per-run replay seeds.

    Notes
    -----
    A SeedSequence child stream per index makes samples independent of worker
    count, completion order, and requested batch length. No process-global RNG.
    """
    if not isinstance(count, int) or count < 1 or not isinstance(seed, int) or seed < 0:
        raise ValueError("Require a positive run count and nonnegative integer seed")
    seeds = tuple(int(child.generate_state(1, dtype=np.uint64)[0]) for child in np.random.SeedSequence(seed).spawn(count))
    unit = np.stack([np.random.default_rng(run_seed).uniform(np.finfo(float).eps, 1.-np.finfo(float).eps,
                                                           len(space.parameters)) for run_seed in seeds])
    return space.transform(unit), seeds


def perturb_launch(vehicle: VehicleConfig, config: SimulationConfig, guidance: GuidanceProgram,
                   space: DispersionSpace, values: ArrayLike) -> tuple[VehicleConfig, SimulationConfig, GuidanceProgram]:
    """Apply SI/fraction perturbations without modifying nominal configuration.

    Notes
    -----
    Vacuum thrust and Isp are independent: constant mdot scales by thrust/Isp.
    Curves scale in force, preserving their time knots. Cutoff schedules and
    geometry remain nominal. Gaussian nonphysical realizations fail explicitly.
    Rail angles may pass smoothly through vertical without reversing the
    commanded pitch program; initial attitude is canonically represented.
    """
    row = np.asarray(values, dtype=float)
    if row.shape != (len(space.parameters),) or not np.all(np.isfinite(row)):
        raise ValueError("One finite input per uncertainty is required")
    known = {f"stage_{i}_{name}_fraction" for i in range(1, len(vehicle.stages)+1)
             for name in ("thrust", "isp", "propellant", "dry")}
    known.update(("azimuth_rad", "elevation_rad", "cd_fraction"))
    known.update(f"wind_{i}_{axis}_m_s" for i in range(len(space.wind_altitude_m)) for axis in ("e", "n", "u"))
    if set(space.names)-known:
        raise ValueError(f"Unknown launch parameters: {sorted(set(space.names)-known)}")
    for p in space.parameters:
        expected = "rad" if p.name.endswith("_rad") else "m/s" if p.name.startswith("wind_") else "1"
        if p.unit != expected:
            raise ValueError(f"Wrong unit for {p.name}: expected {expected}")
    offsets = dict(zip(space.names, row))
    stages = []
    drag_scale = 1.+offsets.get("cd_fraction", 0.)
    for index, stage in enumerate(vehicle.stages, 1):
        thrust_scale = 1.+offsets.get(f"stage_{index}_thrust_fraction", 0.)
        isp_scale = 1.+offsets.get(f"stage_{index}_isp_fraction", 0.)
        if min(thrust_scale, isp_scale) <= 0.:
            raise ValueError("A realized thrust/Isp multiplier is nonpositive")
        changes = dict(isp_vac_s=stage.isp_vac_s*isp_scale,
                       propellant_mass_kg=stage.propellant_mass_kg*(1.+offsets.get(f"stage_{index}_propellant_fraction", 0.)),
                       dry_mass_kg=stage.dry_mass_kg*(1.+offsets.get(f"stage_{index}_dry_fraction", 0.)),
                       aero=replace(stage.aero, cd_multiplier=stage.aero.cd_multiplier*drag_scale))
        if stage.mass_flow_kg_s is not None:
            changes["mass_flow_kg_s"] = stage.mass_flow_kg_s*thrust_scale/isp_scale
        else:
            changes["thrust_curve_vac_n"] = tuple(thrust_scale*np.asarray(stage.thrust_curve_vac_n))
        stages.append(replace(stage, **changes))
    gust_m_s = np.array([[offsets.get(f"wind_{i}_{axis}_m_s", 0.) for axis in ("e", "n", "u")]
                         for i in range(len(space.wind_altitude_m))])
    gust = WindProfile(space.wind_altitude_m, tuple(map(tuple, gust_m_s)))
    knots_m = np.union1d(space.wind_altitude_m, config.wind.altitude_m)
    wind = WindProfile(tuple(knots_m), tuple(map(tuple, config.wind.velocity(knots_m)+gust.velocity(knots_m))))
    daz_rad, del_rad = offsets.get("azimuth_rad", 0.), offsets.get("elevation_rad", 0.)
    steering = replace(guidance.steering, rail_azimuth_rad=guidance.steering.rail_azimuth_rad+daz_rad,
                       rail_elevation_rad=guidance.steering.rail_elevation_rad+del_rad)
    azimuth_rad, elevation_rad = config.initial_azimuth_rad+daz_rad, config.initial_elevation_rad+del_rad
    if elevation_rad > np.pi/2.:
        azimuth_rad, elevation_rad = azimuth_rad+np.pi, np.pi-elevation_rad
    return (replace(vehicle, stages=tuple(stages)),
            replace(config, wind=wind, initial_azimuth_rad=azimuth_rad, initial_elevation_rad=elevation_rad),
            replace(guidance, steering=steering))


@dataclass(frozen=True)
class RunRecord:
    """One indexed realization: replay seed, impact/horizon/failed status, SI metrics, error."""

    index: int
    seed: int
    status: Literal["impact", "horizon", "failed"]
    metrics: TrajectoryMetrics | None
    error: str | None = None


def _run_one(job: tuple[int, int, VehicleConfig, SimulationConfig, GuidanceProgram, DispersionSpace, FloatArray]) -> RunRecord:
    """Execute a picklable isolated launch job and retain numerical/configuration failures."""
    index, seed, vehicle, config, guidance, space, values = job
    try:
        varied_vehicle, varied_config, varied_guidance = perturb_launch(vehicle, config, guidance, space, values)
        result = simulate(varied_vehicle, varied_config, guidance=varied_guidance)
        metrics = trajectory_metrics(result, guidance.steering.rail_azimuth_rad, guidance.steering.target, len(vehicle.stages)-1)
        return RunRecord(index, seed, "impact" if result.termination == "ground_impact" else "horizon", metrics)
    except Exception as error:
        return RunRecord(index, seed, "failed", None, f"{type(error).__name__}: {error}")


def evaluate_design(vehicle: VehicleConfig, config: SimulationConfig, guidance: GuidanceProgram,
                    space: DispersionSpace, inputs: ArrayLike, *, workers: int = 1,
                    seeds: tuple[int, ...] | None = None) -> tuple[RunRecord, ...]:
    """Run a stored (n,d) physical design serially or with Windows-safe spawned processes.

    Notes
    -----
    Results stay in input order, regardless of completion order. Workers return
    summaries only. Exceptions and broken-worker futures become failed records;
    KeyboardInterrupt remains interruptible. Protect parallel callers with main.
    """
    values = np.asarray(inputs, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(space.parameters) or not np.all(np.isfinite(values)) or len(values) == 0:
        raise ValueError("Input design must be finite with shape (n,d), n>0")
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    replay = tuple(range(len(values))) if seeds is None else seeds
    if len(replay) != len(values):
        raise ValueError("One replay seed is required per row")
    jobs = [(i, replay[i], vehicle, config, guidance, space, row) for i, row in enumerate(values)]
    if workers == 1:
        return tuple(map(_run_one, jobs))
    records: list[RunRecord | None] = [None]*len(jobs)
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(_run_one, job): job[0] for job in jobs}
        for future in as_completed(futures):
            index = futures[future]
            try:
                records[index] = future.result()
            except Exception as error:
                records[index] = RunRecord(index, replay[index], "failed", None, f"Worker failure: {type(error).__name__}: {error}")
    return tuple(records)


@dataclass(frozen=True)
class DispersionEllipse:
    """Covariance contour: center [m], covariance [m^2], ordered semiaxes [m].

    Notes
    -----
    Orientation is counterclockwise from downrange [rad]. Coverage is the
    theoretical *bivariate Gaussian* mass at this Mahalanobis radius, not a
    finite-sample tolerance/confidence region or an empirical safety guarantee.
    Rank-deficient covariance returns a degenerate contour and NaN coverage.
    """

    sigma: float
    center_m: FloatArray
    covariance_m2: FloatArray
    semiaxes_m: FloatArray
    orientation_rad: float
    gaussian_coverage: float
    rank: int
    points_m: FloatArray


def covariance_ellipses(impacts_m: ArrayLike, sigmas: tuple[float, ...] = (1., 2., 3.),
                        points: int = 181) -> tuple[DispersionEllipse, ...]:
    """Eigendecompose unbiased sample covariance and return exact scaled contours.

    Parameters
    ----------
    impacts_m : array_like, shape (n,2)
        Finite downrange/crossrange impacts [m]; n >= 2. Missing runs must be
        accounted for by the caller, never silently discarded here.
    sigmas : tuple of float
        Positive Mahalanobis radii (dimensionless).
    points : int
        Number of plotted vertices including the repeated endpoint.
    """
    impacts = np.asarray(impacts_m, dtype=float)
    if impacts.ndim != 2 or impacts.shape[1] != 2 or len(impacts) < 2 or not np.all(np.isfinite(impacts)):
        raise ValueError("At least two finite two-dimensional impacts are required")
    if points < 5 or not sigmas or not np.all(np.isfinite(sigmas)) or min(sigmas) <= 0.:
        raise ValueError("Contour radii and point count must be positive")
    center_m, covariance_m2 = impacts.mean(axis=0), np.cov(impacts, rowvar=False, ddof=1)
    eigenvalues_m2, axes = np.linalg.eigh(covariance_m2)
    eigenvalues_m2, axes = np.maximum(eigenvalues_m2[::-1], 0.), axes[:, ::-1]
    rank = int(np.linalg.matrix_rank(covariance_m2, hermitian=True))
    angle_rad = float(np.arctan2(axes[1, 0], axes[0, 0]))
    theta_rad = np.linspace(0., 2.*np.pi, points)
    circle = np.column_stack((np.cos(theta_rad), np.sin(theta_rad)))
    return tuple(DispersionEllipse(sigma, center_m, covariance_m2, sigma*np.sqrt(eigenvalues_m2), angle_rad,
                                   float(-np.expm1(-.5*sigma**2)) if rank == 2 else np.nan, rank,
                                   center_m+(circle*(sigma*np.sqrt(eigenvalues_m2))) @ axes.T) for sigma in sigmas)


@dataclass(frozen=True)
class MonteCarloResult:
    """Replayable design, ordered runs, conditional SI statistics, contours, metadata."""

    inputs: FloatArray
    input_names: tuple[str, ...]
    runs: tuple[RunRecord, ...]
    statistics: dict[str, dict[str, float | int]]
    ellipses: tuple[DispersionEllipse, ...]
    configuration: dict[str, object]
    metadata: dict[str, object]

    def save(self, path: str | Path) -> None:
        """Export complete nominal configuration, perturbations, seeds, and outcomes as JSON."""
        write_json(path, self)


def run_monte_carlo(vehicle: VehicleConfig, config: SimulationConfig, guidance: GuidanceProgram = GuidanceProgram(), *,
                    count: int = 100, seed: int = 0, workers: int | None = None,
                    space: DispersionSpace | None = None) -> MonteCarloResult:
    """Propagate a seeded ensemble; 100-1000 runs are supported without storing histories.

    Parameters
    ----------
    vehicle, config, guidance : configuration objects
        Nominal SI mission and reusable guidance program.
    count, seed, workers : int
        Ensemble size, root seed, worker count (dimensionless). Default workers
        is min(available CPUs, 8); workers=1 selects serial replay/debugging.
    space : DispersionSpace or None
        None uses documented stage-specific default uncertainties.
    """
    space = default_dispersion_space(len(vehicle.stages)) if space is None else space
    worker_count = min(os.cpu_count() or 1, 8) if workers is None else workers
    inputs, seeds = sample_inputs(space, count, seed)
    records = evaluate_design(vehicle, config, guidance, space, inputs, workers=worker_count, seeds=seeds)
    statistics: dict[str, dict[str, float | int]] = {}
    for name in TrajectoryMetrics.__dataclass_fields__:
        if name == "insertion_achieved":
            continue
        values = np.array([getattr(run.metrics, name) for run in records if run.metrics is not None])
        finite = values[np.isfinite(values)]
        statistics[name] = dict(count=len(finite), missing=count-len(finite), mean=float(finite.mean()) if len(finite) else np.nan,
                                std=float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
                                p025=float(np.quantile(finite, .025)) if len(finite) else np.nan,
                                median=float(np.median(finite)) if len(finite) else np.nan,
                                p975=float(np.quantile(finite, .975)) if len(finite) else np.nan)
    impacts = np.array([[run.metrics.downrange_impact_m, run.metrics.crossrange_impact_m]
                        for run in records if run.status == "impact"])
    ellipses = covariance_ellipses(impacts) if len(impacts) >= 2 else ()
    metadata = dict(provenance(), root_seed=seed, workers=worker_count,
                    status_counts={status: sum(run.status == status for run in records) for status in ("impact", "horizon", "failed")},
                    insertion_successes=sum(run.metrics.insertion_achieved for run in records if run.metrics is not None),
                    projection="spherical azimuthal equidistant; WGS84_A; nominal azimuth; right-positive crossrange")
    return MonteCarloResult(inputs, space.names, records, statistics, ellipses,
                            dict(vehicle=asdict(vehicle), simulation=asdict(config), guidance=asdict(guidance), space=asdict(space)), metadata)


def plot_dispersion(result: MonteCarloResult, path: str | Path) -> None:
    """Save a static SI impact plot with explicitly labeled covariance contours."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(8., 6.))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    impacts_m = np.array([[run.metrics.downrange_impact_m, run.metrics.crossrange_impact_m]
                         for run in result.runs if run.status == "impact"])
    if len(impacts_m):
        axis.scatter(impacts_m[:, 0], impacts_m[:, 1], s=10, alpha=.5, label=f"Observed impacts (n={len(impacts_m)})")
    for ellipse in result.ellipses:
        coverage = f"{ellipse.gaussian_coverage:.1%} Gaussian mass" if ellipse.rank == 2 else "degenerate"
        axis.plot(*ellipse.points_m.T, label=f"{ellipse.sigma:g} sigma: {coverage}")
    axis.set(xlabel="Downrange [m]", ylabel="Right crossrange [m]",
             title=f"Impact covariance contours | {result.metadata['status_counts']}")
    axis.set_aspect("equal", adjustable="datalim")
    if len(impacts_m):
        axis.legend(fontsize=8)
    axis.grid(alpha=.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
