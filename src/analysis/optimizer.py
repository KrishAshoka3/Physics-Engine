"""Constrained two-stage propellant allocation, pitch, and orbital payload sizing.

Notes
-----
Total propellant is fixed; engine thrust, geometry, and Isp are fixed. Dry masses
optionally scale with tank propellant (constant stage structural mass ratio).
This is conceptual sizing, not a tank/engine design or global-optimality proof.
Each result is repropagated with tighter tolerances and half the maximum step.
"""

from dataclasses import asdict, dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import NonlinearConstraint, differential_evolution, minimize

from src.guidance import GuidanceProgram
from src.physics import SimulationConfig, VehicleConfig, simulate
from .metrics import TrajectoryMetrics, provenance, trajectory_metrics

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OptimizationConfig:
    """Bounds, SI structural limits, method, and deterministic evaluation budget.

    Parameters
    ----------
    objective, method : str
        apogee or payload_to_orbit; SLSQP or differential_evolution.
    stage1_fraction_bounds : tuple
        S1 fuel / total fuel bounds (dimensionless, strictly inside (0,1)).
    pitch_kick_bounds_rad : tuple
        Pitch maneuver bounds [rad].
    payload_bounds_kg : tuple
        Additional decision-variable bounds [kg] for orbital payload objective.
    max_q_pa, max_axial_g : float
        Sampled dynamic pressure ceiling [Pa] and absolute axial load [g0].
    scale_dry_mass : bool
        Scale each dry mass in proportion to its allocated fuel mass.
    max_iterations, population_size, max_evaluations, seed : int
        Algorithm budgets, DE population multiplier, and RNG seed (dimensionless).
    """

    objective: Literal["apogee", "payload_to_orbit"] = "apogee"
    method: Literal["SLSQP", "differential_evolution"] = "SLSQP"
    stage1_fraction_bounds: tuple[float, float] = (.2, .9)
    pitch_kick_bounds_rad: tuple[float, float] = (0., np.deg2rad(10.))
    payload_bounds_kg: tuple[float, float] = (0., 1000.)
    max_q_pa: float = 50000.
    max_axial_g: float = 8.
    scale_dry_mass: bool = True
    max_iterations: int = 20
    population_size: int = 5
    max_evaluations: int = 200
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate finite ordered physical bounds, positive limits, and search settings."""
        for bounds in (self.stage1_fraction_bounds, self.pitch_kick_bounds_rad, self.payload_bounds_kg):
            if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
                raise ValueError("Optimization bounds must be finite and strictly ordered")
        if (not 0. < self.stage1_fraction_bounds[0] < self.stage1_fraction_bounds[1] < 1.
                or self.pitch_kick_bounds_rad[0] < 0. or self.payload_bounds_kg[0] < 0.):
            raise ValueError("Nonphysical sizing bounds")
        if not np.all(np.isfinite([self.max_q_pa, self.max_axial_g])) or min(self.max_q_pa, self.max_axial_g) <= 0.:
            raise ValueError("Structural limits must be finite and positive")
        if self.objective not in ("apogee", "payload_to_orbit") or self.method not in ("SLSQP", "differential_evolution"):
            raise ValueError("Unsupported optimization objective or method")
        if any(not isinstance(value, int) or value < 1 for value in (self.max_iterations, self.population_size, self.max_evaluations)):
            raise ValueError("Optimization budgets must be positive integers")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer")


@dataclass(frozen=True)
class DesignPoint:
    """Stage-1 propellant fraction (dimensionless), pitch [rad], and payload [kg]."""

    stage1_propellant_fraction: float
    pitch_kick_rad: float
    payload_mass_kg: float


def sized_vehicle(vehicle: VehicleConfig, design: DesignPoint, *, scale_dry_mass: bool = True) -> VehicleConfig:
    """Allocate a fixed total fuel budget to exactly two stages.

    Notes
    -----
    Constant-flow engine thrust is unchanged. Curve times are stretched in
    proportion to fuel allocation at fixed thrust; explicit cutoff times scale
    likewise. Dry masses optionally keep their baseline dry/fuel ratio. Nozzle
    geometry and aerodynamic dimensions remain fixed provisional assumptions.
    """
    if len(vehicle.stages) != 2 or not 0. < design.stage1_propellant_fraction < 1.:
        raise ValueError("Sizing requires exactly two stages and an interior fuel split")
    if not np.all(np.isfinite([design.stage1_propellant_fraction, design.pitch_kick_rad, design.payload_mass_kg])) or min(design.pitch_kick_rad, design.payload_mass_kg) < 0.:
        raise ValueError("Sizing variables must be finite and nonnegative")
    total_fuel_kg = sum(stage.propellant_mass_kg for stage in vehicle.stages)
    stages = []
    for stage, fraction in zip(vehicle.stages, (design.stage1_propellant_fraction, 1.-design.stage1_propellant_fraction)):
        fuel_kg = total_fuel_kg*fraction
        scale = fuel_kg/stage.propellant_mass_kg
        stages.append(replace(stage, propellant_mass_kg=fuel_kg,
                              dry_mass_kg=stage.dry_mass_kg*(scale if scale_dry_mass else 1.),
                              cutoff_time_s=None if stage.cutoff_time_s is None else scale*stage.cutoff_time_s,
                              thrust_curve_time_s=tuple(scale*np.asarray(stage.thrust_curve_time_s))))
    return replace(vehicle, stages=tuple(stages), payload_mass_kg=design.payload_mass_kg)


@dataclass(frozen=True)
class DesignEvaluation:
    """Actual SI metrics, normalized constraint margins >=0, and objective [m or kg]."""

    design: DesignPoint
    objective_value: float
    metrics: TrajectoryMetrics | None
    constraint_margins: FloatArray
    feasible: bool
    error: str | None = None


@dataclass(frozen=True)
class OptimizationResult:
    """Verified candidate and baseline; feasibility is separate from SciPy convergence.

    Notes
    -----
    ``improved`` compares the repropagated feasible candidate and baseline.
    A budget stop may still return an improved feasible design, but
    ``optimizer_converged`` remains false. No infeasible design is labeled success.
    Search evaluations and full nominal configurations are retained for replay.
    """

    candidate: DesignEvaluation
    baseline: DesignEvaluation
    improved: bool
    optimizer_converged: bool
    message: str
    evaluations: tuple[DesignEvaluation, ...]
    configuration: dict[str, object]
    metadata: dict[str, object]


def evaluate_candidate(vehicle: VehicleConfig, simulation: SimulationConfig, guidance: GuidanceProgram,
                        settings: OptimizationConfig, design: DesignPoint) -> DesignEvaluation:
    """Propagate one design and measure explicit normalized constraint margins.

    Notes
    -----
    Payload must actually meet all insertion radius/radial/speed tolerances at
    final-stage cutoff. Apogee must be observed before the simulation horizon.
    Invalid configurations/integrations become infeasible with recorded errors.
    """
    constraint_count = 5 if settings.objective == "payload_to_orbit" else 3
    try:
        candidate_vehicle = sized_vehicle(vehicle, design, scale_dry_mass=settings.scale_dry_mass)
        candidate_guidance = replace(guidance, steering=replace(guidance.steering, pitch_kick_rad=design.pitch_kick_rad))
        result = simulate(candidate_vehicle, simulation, guidance=candidate_guidance)
        target = guidance.steering.target
        metrics = trajectory_metrics(result, guidance.steering.rail_azimuth_rad, target, 1)
        margins = [1.-metrics.max_q_pa/settings.max_q_pa, 1.-metrics.max_abs_axial_g/settings.max_axial_g]
        if settings.objective == "payload_to_orbit":
            if target is None:
                raise ValueError("Payload optimization requires an orbital target")
            margins.extend((1.-abs(metrics.insertion_radius_error_m)/target.altitude_tolerance_m,
                            1.-abs(metrics.insertion_radial_error_m_s)/target.velocity_tolerance_m_s,
                            1.-abs(metrics.insertion_speed_error_m_s)/target.velocity_tolerance_m_s))
            value = design.payload_mass_kg
        else:
            margins.append(1. if np.isfinite(metrics.apogee_altitude_m) else -1.)
            value = metrics.apogee_altitude_m
        margins_array = np.asarray(margins)
        margins_array[~np.isfinite(margins_array)] = -1e6
        in_bounds = (settings.stage1_fraction_bounds[0] <= design.stage1_propellant_fraction <= settings.stage1_fraction_bounds[1]
                     and settings.pitch_kick_bounds_rad[0] <= design.pitch_kick_rad <= settings.pitch_kick_bounds_rad[1]
                     and (settings.objective != "payload_to_orbit"
                          or settings.payload_bounds_kg[0] <= design.payload_mass_kg <= settings.payload_bounds_kg[1]))
        feasible = bool(in_bounds and np.isfinite(value) and np.all(margins_array >= 0.))
        return DesignEvaluation(design, float(value), metrics, margins_array, feasible)
    except (ValueError, RuntimeError, FloatingPointError) as error:
        return DesignEvaluation(design, np.nan, None, np.full(constraint_count, -1e6), False, f"{type(error).__name__}: {error}")


class _EvaluationBudget(Exception):
    """Internal signal when the dimensionless count of unique search evaluations is exhausted."""


def optimize_trajectory(vehicle: VehicleConfig, simulation: SimulationConfig, guidance: GuidanceProgram,
                        settings: OptimizationConfig = OptimizationConfig()) -> OptimizationResult:
    """Optimize fuel split/pitch, and optionally payload, with explicit nonlinear constraints.

    Parameters
    ----------
    vehicle, simulation, guidance : configuration objects
        Nominal two-stage SI mission. Payload objective requires an OrbitTarget.
    settings : OptimizationConfig
        Physical bounds/limits and deterministic SciPy search budget.

    Returns
    -------
    OptimizationResult
        Best feasible evaluated candidate rechecked at half maximum step and
        tighter rtol, plus rechecked baseline. If no feasible design exists,
        return the baseline marked infeasible and an explicit message.
    """
    if len(vehicle.stages) != 2:
        raise ValueError("Two stages are required for fuel-split optimization")
    if settings.objective == "payload_to_orbit" and guidance.steering.target is None:
        raise ValueError("Payload optimization requires an orbital insertion target")
    if settings.objective == "apogee" and guidance.steering.target is not None:
        raise ValueError("Apogee optimization requires a suborbital guidance program")
    if settings.pitch_kick_bounds_rad[1] >= guidance.steering.rail_elevation_rad:
        raise ValueError("Pitch bounds must keep the kick above the horizon")
    bounds = [settings.stage1_fraction_bounds, settings.pitch_kick_bounds_rad]
    if settings.objective == "payload_to_orbit":
        bounds.append(settings.payload_bounds_kg)
    lower, upper = np.asarray(bounds).T
    baseline_design = DesignPoint(vehicle.stages[0].propellant_mass_kg/sum(s.propellant_mass_kg for s in vehicle.stages),
                                  guidance.steering.pitch_kick_rad, vehicle.payload_mass_kg)
    initial_values = np.array([baseline_design.stage1_propellant_fraction, baseline_design.pitch_kick_rad,
                               baseline_design.payload_mass_kg][:len(bounds)])
    initial = np.clip((initial_values-lower)/(upper-lower), 0., 1.)
    cache: dict[tuple[float, ...], DesignEvaluation] = {}

    def evaluate(normalized: FloatArray) -> DesignEvaluation:
        """Cache exact normalized designs to reuse trajectory runs for objective and constraints."""
        key = tuple(float(value) for value in normalized)
        if key not in cache:
            if len(cache) >= settings.max_evaluations:
                raise _EvaluationBudget()
            physical = lower+np.asarray(normalized)*(upper-lower)
            design = DesignPoint(float(physical[0]), float(physical[1]),
                                  float(physical[2]) if len(physical) == 3 else vehicle.payload_mass_kg)
            cache[key] = evaluate_candidate(vehicle, simulation, guidance, settings, design)
        return cache[key]

    baseline_search = evaluate_candidate(vehicle, simulation, guidance, settings, baseline_design)
    objective_scale = max(abs(baseline_search.objective_value), 1.) if np.isfinite(baseline_search.objective_value) else 1e5

    def objective(normalized: FloatArray) -> float:
        """Return negative scaled altitude/payload objective, with finite invalid-run penalty."""
        value = evaluate(normalized).objective_value
        return -value/objective_scale if np.isfinite(value) else 1e6

    def constraints(normalized: FloatArray) -> FloatArray:
        """Return normalized structural and outcome margins; feasible means all >=0."""
        return evaluate(normalized).constraint_margins

    converged, message = False, ""
    try:
        if settings.method == "SLSQP":
            solved = minimize(objective, initial, method="SLSQP", bounds=[(0., 1.)]*len(bounds),
                              constraints=[dict(type="ineq", fun=constraints)],
                              options=dict(maxiter=settings.max_iterations, ftol=1e-6, eps=1e-3))
        else:
            solved = differential_evolution(objective, [(0., 1.)]*len(bounds), x0=initial,
                                             constraints=NonlinearConstraint(constraints, 0., np.inf),
                                             maxiter=settings.max_iterations, popsize=settings.population_size,
                                             rng=np.random.default_rng(settings.seed), polish=False, workers=1, tol=1e-5)
        converged, message = bool(solved.success), str(solved.message)
    except _EvaluationBudget:
        message = "Unique search evaluation budget exhausted; returning best evaluated feasible design"
    eligible = [evaluation for evaluation in cache.values() if evaluation.feasible]
    baseline_in_bounds = np.all((initial_values >= lower) & (initial_values <= upper))
    if baseline_search.feasible and baseline_in_bounds:
        eligible.append(baseline_search)
    chosen = max(eligible, key=lambda item: item.objective_value) if eligible else baseline_search
    if not eligible:
        message += "; no feasible design found"
    verified_config = replace(simulation, max_step_s=simulation.max_step_s/2., rtol=min(simulation.rtol*.1, 1e-9))
    baseline = evaluate_candidate(vehicle, verified_config, guidance, settings, baseline_design)
    candidate = baseline if chosen.design == baseline_design else evaluate_candidate(vehicle, verified_config, guidance, settings, chosen.design)
    if not eligible:
        candidate = replace(candidate, feasible=False)
    if not candidate.feasible:
        message += "; candidate is infeasible on independent verification"
    improved = bool(candidate.feasible and (not baseline.feasible or candidate.objective_value > baseline.objective_value+1e-6*objective_scale))
    return OptimizationResult(candidate, baseline, improved, converged, message, tuple(cache.values()),
                               dict(vehicle=asdict(vehicle), simulation=asdict(simulation), guidance=asdict(guidance),
                                    optimization=asdict(settings), verification_simulation=asdict(verified_config)),
                               dict(provenance(), search_evaluations=len(cache), seed=settings.seed,
                                    objective_unit="m" if settings.objective == "apogee" else "kg"))
