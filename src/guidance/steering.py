"""Rail, pitch-kick, air-relative gravity turn, and receding-horizon orbital steering.

Notes
-----
Orbital steering is a numerical linear-tangent shooting method, inspired by the
PEG family, not an implementation of flight-certified Shuttle/SLS PEG. It solves
pitch endpoints and burn duration against radius, radial velocity, and tangential
speed using a central-gravity vacuum predictor and explicit feasibility checks.
"""

from dataclasses import dataclass, field
from math import atan, cos, sin, tan
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, least_squares

from src.physics.constants import G0_M_S2, WGS84_A_M
from src.physics.flight import FlightCommand, NavigationState
from src.physics.vehicle import StageConfig, VehicleConfig
from .autopilot import AutopilotConfig, TVCAutopilot

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OrbitTarget:
    """Insertion at perigee: spherical altitude [m], optional apogee [m], vr=0.

    Notes
    -----
    Radius is WGS84 equatorial radius plus altitude, matching orbital telemetry.
    Equal perigee/apogee gives a circular target; inclination is not targeted.
    """

    altitude_m: float = 200000.
    apogee_altitude_m: float | None = None
    altitude_tolerance_m: float = 100.
    velocity_tolerance_m_s: float = 1.

    def __post_init__(self) -> None:
        """Validate positive SI orbit targets and insertion at the lower apsis."""
        values = (self.altitude_m, self.altitude_tolerance_m, self.velocity_tolerance_m_s)
        if not np.all(np.isfinite(values)) or min(values) <= 0.:
            raise ValueError("Target altitude and tolerances must be finite and positive")
        if self.apogee_altitude_m is not None and (not np.isfinite(self.apogee_altitude_m) or self.apogee_altitude_m < self.altitude_m):
            raise ValueError("Target apogee must be finite and at least the insertion altitude")

    def speed_m_s(self, mu_m3_s2: float) -> float:
        """Return target perigee speed [m/s] from the vis-viva equation."""
        if mu_m3_s2 <= 0. or not np.isfinite(mu_m3_s2):
            raise ValueError("Orbital guidance requires a positive gravitational parameter")
        radius_m = WGS84_A_M + self.altitude_m
        apogee_m = WGS84_A_M + (self.altitude_m if self.apogee_altitude_m is None else self.apogee_altitude_m)
        return float(np.sqrt(mu_m3_s2*(2./radius_m - 2./(radius_m+apogee_m))))


@dataclass(frozen=True)
class SteeringConfig:
    """Multi-phase program in SI units; all angles in radians.

    Parameters
    ----------
    rail_length_m, rail_azimuth_rad, rail_elevation_rad : float
        Rail length [m], azimuth east of north [rad], elevation [rad] in (0,pi).
        Elevations just above pi/2 represent tilt errors through vertical while
        retaining the pitch-program azimuth. Zero length disables the rail.
    vertical_hold_s, pitch_duration_s, pitch_kick_rad : float
        Hold after rail exit [s], kick duration [s], and downward pitch change [rad].
    pitch_profile : {"linear", "exponential"}
        Open-loop ramp; exponential uses a normalized smooth exponential easing.
    guidance_period_s : float
        Orbital replanning/ideal-mode sample interval [s].
    terminal_guidance_s : float
        In 6DoF, freeze the pitch law and trim only cutoff during the final
        interval [s], avoiding ill-conditioned, untrackable last-second turns.
    orbital_stage_index : int
        First stage to use orbital steering (zero-based, dimensionless).
    target : OrbitTarget or None
        None selects a sounding/gravity-turn program.
    """

    rail_length_m: float = 15.
    rail_azimuth_rad: float = np.pi/2.
    rail_elevation_rad: float = np.pi/2.
    vertical_hold_s: float = 5.
    pitch_duration_s: float = 10.
    pitch_kick_rad: float = np.deg2rad(5.)
    pitch_profile: Literal["linear", "exponential"] = "exponential"
    guidance_period_s: float = 5.
    terminal_guidance_s: float = 5.
    orbital_stage_index: int = 1
    target: OrbitTarget | None = None

    def __post_init__(self) -> None:
        """Validate physical ranges and finite configuration quantities."""
        values = (self.rail_length_m, self.rail_azimuth_rad, self.rail_elevation_rad,
                  self.vertical_hold_s, self.pitch_duration_s, self.pitch_kick_rad, self.guidance_period_s, self.terminal_guidance_s)
        if not np.all(np.isfinite(values)) or min(self.rail_length_m, self.vertical_hold_s, self.pitch_kick_rad) < 0.:
            raise ValueError("Rail, hold, and pitch kick must be finite and nonnegative")
        if min(self.pitch_duration_s, self.guidance_period_s, self.terminal_guidance_s) <= 0. or not 0. < self.rail_elevation_rad < np.pi:
            raise ValueError("Require positive durations and an upward rail")
        if self.pitch_kick_rad >= self.rail_elevation_rad or self.pitch_profile not in ("linear", "exponential"):
            raise ValueError("Pitch kick must end above the horizon and use a supported profile")
        if not isinstance(self.orbital_stage_index, int) or self.orbital_stage_index < 0:
            raise ValueError("Orbital stage index must be a nonnegative integer")

    @property
    def rail_axis_enu(self) -> FloatArray:
        """Return rail east/north/up unit direction (dimensionless)."""
        return np.array([cos(self.rail_elevation_rad)*sin(self.rail_azimuth_rad),
                         cos(self.rail_elevation_rad)*cos(self.rail_azimuth_rad), sin(self.rail_elevation_rad)])


def pitch_kick(time_s: float, start_s: float, duration_s: float, initial_pitch_rad: float,
               kick_rad: float, profile: Literal["linear", "exponential"] = "exponential") -> float:
    """Return commanded elevation [rad] for a bounded open-loop pitch maneuver.

    Notes
    -----
    The exponential easing has zero endpoint slopes. Clipping is confined to a
    command profile, never trajectory state or a physical cutoff/rail event.
    """
    if (not np.all(np.isfinite([time_s, start_s, duration_s, initial_pitch_rad, kick_rad]))
            or duration_s <= 0. or profile not in ("linear", "exponential")):
        raise ValueError("Pitch duration must be positive and profile supported")
    fraction = float(np.clip((time_s-start_s)/duration_s, 0., 1.))
    if profile == "exponential":
        eased = (1.-np.exp(-4.*fraction))/(1.-np.exp(-4.))
        fraction = eased*eased*(3.-2.*eased)
    return initial_pitch_rad - kick_rad*fraction


def gravity_turn_direction(relative_velocity_eci_m_s: FloatArray, fallback_axis: FloatArray) -> FloatArray:
    """Return alpha=0 air-relative axis; use a unit fallback at speed below 1e-6 m/s."""
    if (np.shape(relative_velocity_eci_m_s) != (3,) or np.shape(fallback_axis) != (3,)
            or not np.all(np.isfinite(relative_velocity_eci_m_s)) or not np.all(np.isfinite(fallback_axis))):
        raise ValueError("Gravity turn requires finite three-vectors")
    speed_m_s = float(np.linalg.norm(relative_velocity_eci_m_s))
    if speed_m_s > 1e-6:
        return relative_velocity_eci_m_s/speed_m_s
    if not np.isclose(np.linalg.norm(fallback_axis), 1., atol=1e-8):
        raise ValueError("Zero-speed fallback must be a unit direction")
    return fallback_axis.copy()


def available_burn_s(stage: StageConfig, elapsed_s: float, fuel_kg: float) -> float:
    """Return usable remaining burn [s], respecting scheduled cutoff and curve fuel integral."""
    remaining_s = max(0., stage.nominal_cutoff_s-elapsed_s)
    if fuel_kg <= 0. or remaining_s == 0.:
        return 0.
    if stage.mass_flow_kg_s is not None:
        return min(remaining_s, fuel_kg/stage.mass_flow_kg_s)
    internal_s = np.asarray(stage.thrust_curve_time_s)
    times_s = np.concatenate(([elapsed_s], internal_s[(internal_s > elapsed_s) & (internal_s < stage.nominal_cutoff_s)], [stage.nominal_cutoff_s]))
    flow_kg_s = np.interp(times_s, stage.thrust_curve_time_s, stage.thrust_curve_vac_n)/(stage.isp_vac_s*G0_M_S2)
    consumed_kg = np.concatenate(([0.], np.cumsum(.5*(flow_kg_s[:-1]+flow_kg_s[1:])*np.diff(times_s))))
    if consumed_kg[-1] <= fuel_kg:
        return remaining_s
    index = int(np.searchsorted(consumed_kg, fuel_kg, side="right")-1)
    interval_s = times_s[index+1]-times_s[index]
    slope_kg_s2 = (flow_kg_s[index+1]-flow_kg_s[index])/interval_s
    root_s = brentq(lambda dt_s: consumed_kg[index]+flow_kg_s[index]*dt_s+.5*slope_kg_s2*dt_s**2-fuel_kg, 0., interval_s)
    return float(times_s[index]+root_s-elapsed_s)


@dataclass(frozen=True)
class LinearTangentSolution:
    """Solved pitch endpoints [rad], epoch/burn [s], terminal errors [m,m/s,m/s]."""

    epoch_s: float
    duration_s: float
    initial_pitch_rad: float
    final_pitch_rad: float
    errors: FloatArray
    feasible: bool
    evaluations: int

    def pitch_rad(self, time_s: float) -> float:
        """Evaluate atan(A+B*t) pitch [rad] for mission elapsed time [s]."""
        fraction = np.clip((time_s-self.epoch_s)/self.duration_s, 0., 1.)
        return float(np.arctan(tan(self.initial_pitch_rad)+(tan(self.final_pitch_rad)-tan(self.initial_pitch_rad))*fraction))

    def pitch_rate_rad_s(self, time_s: float) -> float:
        """Return analytic pitch rate [rad/s] within the planned burn interval."""
        slope_per_s = (tan(self.final_pitch_rad)-tan(self.initial_pitch_rad))/self.duration_s
        return slope_per_s*cos(self.pitch_rad(time_s))**2

    def direction(self, navigation: NavigationState) -> FloatArray:
        """Resolve current linear-tangent pitch into an ECI prograde unit vector."""
        radial = navigation.position_eci_m/np.linalg.norm(navigation.position_eci_m)
        tangent_m_s = navigation.velocity_eci_m_s - np.dot(navigation.velocity_eci_m_s, radial)*radial
        tangent = gravity_turn_direction(tangent_m_s, navigation.local_to_eci[:, 0])
        pitch_rad = self.pitch_rad(navigation.time_s)
        return cos(pitch_rad)*tangent + sin(pitch_rad)*radial


def solve_linear_tangent(navigation: NavigationState, stage: StageConfig, target: OrbitTarget,
                         previous: LinearTangentSolution | None = None, *, freeze_pitch: bool = False) -> LinearTangentSolution:
    """Shoot pitch, pitch rate, and cutoff time to an orbital insertion target.

    Parameters
    ----------
    navigation : NavigationState
        Current ECI truth/estimate with mass [kg], position [m], velocity [m/s].
    stage : StageConfig
        Active stage thrust/flow inputs; predictor assumes vacuum and no staging.
    target : OrbitTarget
        Perigee altitude [m] and circular/elliptical speed constraint [m/s].
    previous : LinearTangentSolution or None
        Warm start from the previous accepted guidance update.
    freeze_pitch : bool
        Hold the previous tangent slope/end direction and solve only cutoff.
        Used near 6DoF insertion to avoid demanding instantaneous attitude changes.

    Returns
    -------
    LinearTangentSolution
        Bounded numerical plan and explicit feasibility; an infeasible plan must
        not be silently treated as a successful orbital insertion.
    """
    radius_m = float(np.linalg.norm(navigation.position_eci_m))
    radial_m_s = float(np.dot(navigation.position_eci_m, navigation.velocity_eci_m_s)/radius_m)
    tangent_m_s = float(np.linalg.norm(navigation.velocity_eci_m_s-radial_m_s*navigation.position_eci_m/radius_m))
    target_radius_m = WGS84_A_M + target.altitude_m
    target_speed_m_s = target.speed_m_s(navigation.mu_m3_s2)
    remaining_s = available_burn_s(stage, navigation.stage_elapsed_s, navigation.propellant_kg)
    if remaining_s <= .05 or not navigation.engine_on:
        return LinearTangentSolution(navigation.time_s, max(remaining_s, .001), 0., 0., np.array([radius_m-target_radius_m, radial_m_s, tangent_m_s-target_speed_m_s]), False, 0)
    initial = np.array([radius_m, radial_m_s, tangent_m_s, navigation.mass_kg])

    def terminal(parameters: FloatArray) -> FloatArray:
        """Predict terminal radius [m], vr/vt [m/s], and mass [kg] for bounded parameters."""
        pitch0_rad, pitch1_rad, duration_s = parameters
        tangent0, tangent_delta = tan(pitch0_rad), tan(pitch1_rad)-tan(pitch0_rad)

        def derivative(elapsed_s: float, state: FloatArray) -> tuple[float, float, float, float]:
            """Return radial predictor derivatives [m/s,m/s^2,m/s^2,kg/s]."""
            distance_m, vr_m_s, vt_m_s, mass_kg = state
            flow_kg_s = (stage.mass_flow_kg_s if stage.mass_flow_kg_s is not None else
                         float(np.interp(navigation.stage_elapsed_s+elapsed_s, stage.thrust_curve_time_s,
                                         stage.thrust_curve_vac_n))/(stage.isp_vac_s*G0_M_S2))
            acceleration_m_s2 = flow_kg_s*stage.isp_vac_s*G0_M_S2/mass_kg
            pitch_rad = atan(tangent0+tangent_delta*elapsed_s/duration_s)
            return (vr_m_s, acceleration_m_s2*sin(pitch_rad)+vt_m_s**2/distance_m-navigation.mu_m3_s2/distance_m**2,
                    acceleration_m_s2*cos(pitch_rad)-vr_m_s*vt_m_s/distance_m, -flow_kg_s)

        result = solve_ivp(derivative, (0., duration_s), initial, method="DOP853", rtol=2e-8,
                           atol=[.001, 1e-5, 1e-5, 1e-7], max_step=max(duration_s/16., .01))
        if not result.success:
            raise RuntimeError("Orbital guidance predictor failed: " + result.message)
        return result.y[:, -1]

    scale = np.array([10000., 1000., 1000.])

    def residual(parameters: FloatArray) -> FloatArray:
        """Return dimensionless scaled radius/radial/tangential terminal residuals."""
        return (terminal(parameters)[:3]-[target_radius_m, 0., target_speed_m_s])/scale

    if freeze_pitch and previous is not None:
        # Continue the same tangent profile in absolute time. Adjusting burn
        # duration must not reset the pitch/rate and kick the attitude controller.
        pitch0_rad = previous.pitch_rad(navigation.time_s)
        slope_per_s = (tan(previous.final_pitch_rad)-tan(previous.initial_pitch_rad))/previous.duration_s

        def frozen_parameters(duration_s: float) -> FloatArray:
            """Return continuous pitch endpoints [rad] and candidate remaining burn [s]."""
            return np.array([pitch0_rad, atan(tan(pitch0_rad)+slope_per_s*duration_s), duration_s])

        remaining_plan_s = previous.epoch_s+previous.duration_s-navigation.time_s
        upper_s = min(remaining_s, max(.02, remaining_plan_s+2.))

        def speed_error_m_s(duration_s: float) -> float:
            """Return predicted tangential insertion-speed error [m/s]."""
            return float(terminal(frozen_parameters(duration_s))[2]-target_speed_m_s)

        low_error_m_s, high_error_m_s = speed_error_m_s(.01), speed_error_m_s(upper_s)
        if low_error_m_s*high_error_m_s <= 0.:
            duration_s, root = brentq(speed_error_m_s, .01, upper_s, xtol=1e-8, full_output=True)
            parameters = frozen_parameters(duration_s)
            errors = residual(parameters)*scale
            feasible = bool(abs(errors[0]) <= target.altitude_tolerance_m
                            and np.max(np.abs(errors[1:])) <= target.velocity_tolerance_m_s)
            return LinearTangentSolution(navigation.time_s, duration_s, float(parameters[0]), float(parameters[1]),
                                         errors, feasible, root.function_calls+2)
        errors = residual(frozen_parameters(max(.01, min(upper_s, remaining_plan_s))))*scale
        return LinearTangentSolution(navigation.time_s, max(.01, min(upper_s, remaining_plan_s)),
                                     pitch0_rad, previous.final_pitch_rad, errors, False, 3)

    bounds = ([-1.45, -1.45, .01], [1.45, 1.45, remaining_s])
    guess = np.array([.2, -.1, .85*remaining_s])
    if previous is not None and previous.epoch_s+previous.duration_s > navigation.time_s:
        guess = np.array([previous.pitch_rad(navigation.time_s), previous.final_pitch_rad,
                          previous.epoch_s+previous.duration_s-navigation.time_s])
    guess = np.clip(guess, np.array(bounds[0])+1e-8, np.array(bounds[1])-1e-8)
    result = least_squares(residual, guess, bounds=bounds, x_scale=[.5, .5, max(remaining_s, 1.)],
                           ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=45)
    errors = residual(result.x)*scale
    feasible = bool(abs(errors[0]) <= target.altitude_tolerance_m and np.max(np.abs(errors[1:])) <= target.velocity_tolerance_m_s)
    return LinearTangentSolution(navigation.time_s, float(result.x[2]), float(result.x[0]), float(result.x[1]), errors, feasible, result.nfev)


@dataclass(frozen=True)
class _OpenLoopLaw:
    """Immutable phase law; time [s], angular commands [rad], axes dimensionless."""

    config: SteeringConfig
    phase: str
    kick_start_s: float

    def direction(self, navigation: NavigationState) -> FloatArray:
        """Evaluate a pure ECI nose-axis command without changing controller memory."""
        rail = navigation.launch_to_eci @ self.config.rail_axis_enu
        if self.phase in ("rail", "vertical_hold"):
            return rail
        if self.phase == "pitch_kick":
            pitch_rad = pitch_kick(navigation.time_s, self.kick_start_s, self.config.pitch_duration_s,
                                   self.config.rail_elevation_rad, self.config.pitch_kick_rad, self.config.pitch_profile)
            horizontal = (sin(self.config.rail_azimuth_rad)*navigation.local_to_eci[:, 0]
                          + cos(self.config.rail_azimuth_rad)*navigation.local_to_eci[:, 1])
            return cos(pitch_rad)*horizontal+sin(pitch_rad)*navigation.local_to_eci[:, 2]
        return gravity_turn_direction(navigation.relative_velocity_eci_m_s, rail)


@dataclass(frozen=True)
class GuidanceProgram:
    """Reusable guidance factory for ideal 3DoF or sampled 6DoF TVC tracking.

    Parameters
    ----------
    steering : SteeringConfig
        Phase timing, rail geometry, and optional orbital target.
    autopilot : AutopilotConfig
        Physical TVC controller/servo settings, used only in 6DoF.
    """

    steering: SteeringConfig = field(default_factory=SteeringConfig)
    autopilot: AutopilotConfig = field(default_factory=AutopilotConfig)

    def create_session(self, vehicle: VehicleConfig, dof: int) -> "GuidanceSession":
        """Create isolated per-run memory using vehicle SI inputs and a DoF selector."""
        return GuidanceSession(vehicle, dof, self.steering, self.autopilot)


class GuidanceSession:
    """Event-aware sampled flight computer; all mutable memory is run-local."""

    def __init__(self, vehicle: VehicleConfig, dof: int, config: SteeringConfig, autopilot: AutopilotConfig) -> None:
        """Initialize phase clocks [s], optional orbital planner, and TVC controller."""
        if dof not in (3, 6):
            raise ValueError("Guidance supports 3DoF or 6DoF")
        if dof == 6:
            for stage in vehicle.stages:
                if stage.thrust_offset_body_m[0] >= 0. or np.linalg.norm(stage.thrust_offset_body_m[1:]) > 1e-12:
                    raise ValueError("TVC requires an explicit aft axial nozzle offset on every stage")
        self.vehicle: VehicleConfig = vehicle
        self.dof: int = dof
        self.config: SteeringConfig = config
        self.controller: TVCAutopilot = TVCAutopilot(autopilot)
        self.rail_length_m: float = config.rail_length_m
        self.rail_axis_enu: FloatArray = config.rail_axis_enu
        self.actuator_time_constant_s: float = autopilot.actuator_time_constant_s
        self.actuator_rate_limit_rad_s: float = autopilot.gimbal_rate_limit_rad_s
        self.actuator_angle_limit_rad: float = autopilot.gimbal_limit_rad
        self.release_time_s: float | None = 0. if self.rail_length_m == 0. else None
        self.last_plan_s: float = -np.inf
        self.next_plan_s: float = -np.inf
        self.solution: LinearTangentSolution | None = None
        self.last_attempt: LinearTangentSolution | None = None
        self.last_stage_index: int = -1
        self.records: list[dict[str, object]] = []

    def update(self, navigation: NavigationState, event: str | None = None) -> FlightCommand:
        """Update phases/planning/PID only at accepted sample times or physical events.

        Returns
        -------
        FlightCommand
            Pure direction law, held gimbal targets [rad], and future event clocks [s].
        """
        time_s = navigation.time_s
        if navigation.stage_index != self.last_stage_index:
            self.solution, self.last_attempt = None, None
            self.last_plan_s = -np.inf
            self.next_plan_s = -np.inf
            self.last_stage_index = navigation.stage_index
        if event == "rail_release":
            self.release_time_s = time_s
        if event in ("rail_release", "await_separation", "await_ignition", "guided_cutoff", "propellant_depletion", "scheduled_cutoff"):
            self.controller.reset()
        if (event in ("guided_cutoff", "propellant_depletion", "scheduled_cutoff")
                and self.config.target is not None and navigation.stage_index >= self.config.orbital_stage_index):
            radius_m = float(np.linalg.norm(navigation.position_eci_m))
            radial_m_s = float(np.dot(navigation.velocity_eci_m_s, navigation.position_eci_m)/radius_m)
            tangential_m_s = float(np.linalg.norm(navigation.velocity_eci_m_s-radial_m_s*navigation.position_eci_m/radius_m))
            errors = np.array([radius_m-WGS84_A_M-self.config.target.altitude_m, radial_m_s,
                               tangential_m_s-self.config.target.speed_m_s(navigation.mu_m3_s2)])
            achieved = bool(abs(errors[0]) <= self.config.target.altitude_tolerance_m
                            and np.max(np.abs(errors[1:])) <= self.config.target.velocity_tolerance_m_s)
            self.records.append({"time_s": time_s, "phase": "insertion", "achieved": achieved,
                                 "errors_m_m_s_m_s": errors.tolist(), "cutoff_trigger": event})
        kick_start_s = (self.release_time_s if self.release_time_s is not None else time_s) + self.config.vertical_hold_s
        kick_end_s = kick_start_s+self.config.pitch_duration_s
        phase = ("rail" if navigation.rail_attached else "vertical_hold" if time_s < kick_start_s-1e-9
                 else "pitch_kick" if time_s < kick_end_s-1e-9 else "gravity_turn")
        orbital = (self.config.target is not None and navigation.stage_index >= self.config.orbital_stage_index
                   and navigation.engine_on and not navigation.rail_attached)
        if orbital:
            phase = "orbital"
            if time_s >= self.next_plan_s-1e-9 or event in ("await_ignition", "fairing_jettison"):
                freeze_pitch = (self.dof == 6 and self.solution is not None
                                and self.solution.epoch_s+self.solution.duration_s-time_s <= self.config.terminal_guidance_s)
                self.last_attempt = solve_linear_tangent(navigation, self.vehicle.stages[navigation.stage_index], self.config.target,
                                                         self.solution, freeze_pitch=freeze_pitch)
                self.last_plan_s = time_s
                if self.last_attempt.feasible:
                    self.solution = self.last_attempt
                remaining_plan_s = (self.solution.epoch_s+self.solution.duration_s-time_s
                                    if self.solution is not None else self.config.guidance_period_s*5.)
                interval_s = (min(self.config.guidance_period_s, max(.1, .2*remaining_plan_s))
                              if self.dof == 3 or freeze_pitch else self.config.guidance_period_s)
                if self.dof == 6 and not freeze_pitch and remaining_plan_s > self.config.terminal_guidance_s:
                    interval_s = min(interval_s, remaining_plan_s-self.config.terminal_guidance_s)
                self.next_plan_s = time_s+max(.1, interval_s)
                self.records.append({"time_s": time_s, "phase": "orbital_plan", "feasible": self.last_attempt.feasible,
                                     "errors_m_m_s_m_s": self.last_attempt.errors.tolist(), "evaluations": self.last_attempt.evaluations,
                                     "initial_pitch_rad": self.last_attempt.initial_pitch_rad,
                                     "initial_pitch_rate_rad_s": self.last_attempt.pitch_rate_rad_s(time_s),
                                     "burn_duration_s": self.last_attempt.duration_s, "pitch_frozen": freeze_pitch})
        law = _OpenLoopLaw(self.config, phase, kick_start_s).direction
        cutoff_s = np.inf
        if orbital and self.solution is not None:
            law = self.solution.direction
            cutoff_s = self.solution.epoch_s+self.solution.duration_s
        elif orbital:
            phase = "orbital_infeasible"
        if not navigation.engine_on and navigation.stage_index == len(self.vehicle.stages)-1:
            phase = "coast"
        period_s = self.controller.config.update_period_s if self.dof == 6 else self.config.guidance_period_s
        # Integer clock ticks avoid step-size-dependent controller timing drift.
        next_update_s = (np.floor((time_s+1e-9)/period_s)+1.)*period_s
        if orbital and self.next_plan_s > time_s+1e-9:
            next_update_s = min(next_update_s, self.next_plan_s)
        if not navigation.rail_attached and not orbital:
            for boundary_s in (kick_start_s, kick_end_s):
                if boundary_s > time_s+1e-9:
                    next_update_s = min(next_update_s, boundary_s)
        if phase == "coast":
            # No final-stage TVC authority: continuous servo dynamics and all
            # physical events remain active, but idle sample clocks are unnecessary.
            next_update_s = np.inf
        target_rad = np.zeros(2)
        if self.dof == 6 and not navigation.rail_attached:
            stage = self.vehicle.stages[navigation.stage_index]
            target_rad = self.controller.command(time_s, navigation.body_to_eci, navigation.body_rate_rad_s,
                                                 law(navigation), navigation.inertia_kg_m2,
                                                 -navigation.mass_flow_kg_s*np.asarray(stage.principal_inertia_per_mass_m2),
                                                 navigation.thrust_n, -stage.thrust_offset_body_m[0], navigation.gimbal_rad)
        self.records.append({"time_s": time_s, "phase": phase, "event": event, "gimbal_target_rad": target_rad.tolist(),
                             "saturated": self.controller.saturated})
        return FlightCommand(law, float(next_update_s), float(cutoff_s), target_rad, phase, navigation.rail_attached)
