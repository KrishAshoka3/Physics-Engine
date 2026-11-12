"""Event-driven 3DoF/6DoF propagation in ECI or launch-site rotating ENU.

Notes
-----
State columns are [position (3, m), velocity (3, m/s), total mass (kg), active
stage propellant (kg)], followed in 6DoF by [q_wxyz (4, dimensionless), inertial
body angular velocity (3, rad/s)]. Guided 6DoF appends actual pitch/yaw gimbal
angles (2, rad), giving 17 columns. ENU position is relative to the launch site;
ECI position is geocentric. See ``docs/physics_models.md`` for event semantics,
open-system rotational assumptions, and diagnostic orbital-element conventions.
"""

from dataclasses import dataclass, field, replace
from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from .aerodynamics import WindProfile, aerodynamic_force
from .atmosphere import atmosphere
from .constants import EARTH_J2, EARTH_MU_M3_S2, EARTH_OMEGA_RAD_S, G0_M_S2, WGS84_A_M
from .frames import (ecef_to_geodetic, enu_basis, geodetic_to_ecef, gravity_acceleration,
                     quaternion_derivative, quaternion_matrix, rotation_z)
from .propulsion import propulsion
from .vehicle import VehicleConfig
from .flight import FlightCommand, FlightProgram, FlightSession, NavigationState, gimbal_direction

FloatArray = NDArray[np.float64]
DirectionFunction = Callable[[float, FloatArray], FloatArray]


@dataclass(frozen=True)
class ControlInput:
    """Instantaneous 6DoF control: body thrust unit vector and applied torque [N*m].

    Notes
    -----
    Legacy pure control callback input. Body +x is the nominal thrust/nose axis.
    """

    thrust_direction_body: tuple[float, float, float] = (1., 0., 0.)
    torque_body_n_m: tuple[float, float, float] = (0., 0., 0.)

    def __post_init__(self) -> None:
        """Validate finite torque [N*m] and dimensionless unit thrust direction."""
        if (np.shape(self.thrust_direction_body) != (3,) or np.shape(self.torque_body_n_m) != (3,)
                or not np.all(np.isfinite(self.thrust_direction_body)) or not np.all(np.isfinite(self.torque_body_n_m))
                or not np.isclose(np.linalg.norm(self.thrust_direction_body), 1., atol=1e-8)):
            raise ValueError("Control requires a finite unit thrust direction and three-component torque")


ControlFunction = Callable[[float, FloatArray], ControlInput]
_DEFAULT_CONTROL: ControlInput = ControlInput()


@dataclass(frozen=True)
class SimulationConfig:
    """Propagation inputs and tolerances with explicit physical units.

    Parameters
    ----------
    duration_s, max_step_s : float
        Mission horizon and maximum solver step [s]. Events remain root-located.
    frame, dof, method : str or int
        Dimensionless selectors: ECI/ENU, 3/6, DOP853/RK45.
    rotating_earth, atmosphere_enabled, j2_enabled, engines_enabled : bool
        Model switches (not physical quantities).
    latitude_rad, longitude_rad, launch_altitude_m : float
        WGS84 launch-site coordinates [rad, rad, m].
    initial_velocity_enu_m_s : tuple
        Initial Earth-relative local ENU velocity [m/s].
    initial_azimuth_rad, initial_elevation_rad : float
        6DoF initial body-axis azimuth east of north and elevation above horizon
        [rad]. 3DoF uses the direction callback (local up by default).
    initial_body_rate_rad_s : tuple
        Body angular velocity relative to inertial space, in body axes [rad/s].
    ground_altitude_m : float
        Constant geodetic impact surface [m]; defaults to the WGS84 ellipsoid.
    mu_m3_s2 : float
        Earth gravity parameter [m^3/s^2], zero for force-isolation benchmarks.
    rtol : float
        Dimensionless relative tolerance.
    position_atol_m, velocity_atol_m_s, mass_atol_kg : float
        Absolute integration tolerances in state units.
    quaternion_atol, rate_atol_rad_s : float
        Quaternion (dimensionless) and rate [rad/s] absolute tolerances.
    wind : WindProfile
        Height-dependent local geographic ENU wind [m/s].
    """

    duration_s: float = 1000.
    max_step_s: float = 2.
    frame: Literal["ECI", "ENU"] = "ECI"
    dof: Literal[3, 6] = 3
    method: Literal["DOP853", "RK45"] = "DOP853"
    rotating_earth: bool = True
    atmosphere_enabled: bool = True
    j2_enabled: bool = True
    engines_enabled: bool = True
    latitude_rad: float = 0.
    longitude_rad: float = 0.
    launch_altitude_m: float = 0.
    initial_velocity_enu_m_s: tuple[float, float, float] = (0., 0., 0.)
    initial_azimuth_rad: float = np.pi / 2.
    initial_elevation_rad: float = np.pi / 2.
    initial_body_rate_rad_s: tuple[float, float, float] = (0., 0., 0.)
    ground_altitude_m: float = 0.
    mu_m3_s2: float = EARTH_MU_M3_S2
    rtol: float = 1e-10
    position_atol_m: float = 1e-4
    velocity_atol_m_s: float = 1e-6
    mass_atol_kg: float = 1e-7
    quaternion_atol: float = 1e-11
    rate_atol_rad_s: float = 1e-10
    wind: WindProfile = field(default_factory=WindProfile)

    def __post_init__(self) -> None:
        """Reject unsupported selectors and nonfinite/nonphysical SI inputs."""
        if self.frame not in ("ECI", "ENU") or self.dof not in (3, 6) or self.method not in ("DOP853", "RK45"):
            raise ValueError("Unsupported propagation frame, degrees of freedom, or integration method")
        positive = (self.duration_s, self.max_step_s, self.rtol, self.position_atol_m,
                    self.velocity_atol_m_s, self.mass_atol_kg, self.quaternion_atol, self.rate_atol_rad_s)
        coordinates = (self.latitude_rad, self.longitude_rad, self.launch_altitude_m,
                       self.initial_azimuth_rad, self.initial_elevation_rad, self.ground_altitude_m, self.mu_m3_s2)
        if not np.all(np.isfinite(positive + coordinates)) or min(positive) <= 0. or self.mu_m3_s2 < 0.:
            raise ValueError("Simulation inputs must be finite and tolerances/durations positive")
        if abs(self.latitude_rad) > np.pi / 2. or abs(self.initial_elevation_rad) > np.pi / 2.:
            raise ValueError("Latitude and elevation must lie in [-pi/2, pi/2]")
        if self.launch_altitude_m < self.ground_altitude_m or self.ground_altitude_m < -1000.:
            raise ValueError("Launch must not start below ground; modeled ground must be >= -1000 m")
        for vector in (self.initial_velocity_enu_m_s, self.initial_body_rate_rad_s):
            if np.shape(vector) != (3,) or not np.all(np.isfinite(vector)):
                raise ValueError("Initial velocity and body rate require finite three-vectors")


@dataclass(frozen=True)
class EventRecord:
    """Terminal event with time [s], zero-based stage index, and pre/post SI states.

    Notes
    -----
    Position, velocity, attitude, and rate remain continuous (no separation
    impulse). Mass and active-stage fuel may jump. ``trigger`` identifies the
    root function, e.g. depletion versus scheduled cutoff.
    """

    name: str
    trigger: str
    time_s: float
    stage_index: int
    state_before: FloatArray
    state_after: FloatArray


@dataclass(frozen=True)
class SimulationResult:
    """Strictly increasing time [s], right-continuous state history, events, telemetry.

    Attributes
    ----------
    time_s, state : ndarray
        Samples [s] and (sample, state-component) SI states in ``config.frame``.
    stage_index, engine_on : ndarray
        Zero-based stage and Boolean engine mode (dimensionless).
    events : tuple of EventRecord
        Ordered terminal-event transitions, including both sides of mass jumps.
    telemetry : dict
        Unit-bearing columns; ``mach``, ``twr``, ``axial_g``, ``eccentricity``
        are dimensionless. Undefined orbital diagnostics are NaN, not fabricated.
    config : SimulationConfig
        Frame, epoch, model switches, and tolerances used.
    termination : str
        ``ground_impact`` or ``duration`` (not a physical quantity).
    function_evaluations : int
        Solver RHS call count (dimensionless).
    guidance_records : tuple of dict
        Accepted guidance/control updates, planner feasibility, and actual
        insertion errors with units in keys; empty without a flight program.
    """

    time_s: FloatArray
    state: FloatArray
    stage_index: NDArray[np.int64]
    engine_on: NDArray[np.bool_]
    events: tuple[EventRecord, ...]
    telemetry: dict[str, FloatArray]
    config: SimulationConfig
    termination: str
    function_evaluations: int
    guidance_records: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True)
class _Mode:
    """Discrete phase, stage index, ignition epoch [s], and next event time [s]."""

    phase: Literal["powered", "await_separation", "await_ignition", "coast"]
    stage_index: int = 0
    ignition_s: float = 0.
    next_time_s: float = np.inf


@dataclass
class _TerminalEvent:
    """Root function with units specified by its name; direction is dimensionless."""

    name: str
    function: Callable[[float, FloatArray], float]
    direction: float = 1.
    terminal: bool = True

    def __call__(self, time_s: float, state: FloatArray) -> float:
        """Evaluate continuous event residual [s, m, or kg] at time [s] and SI state."""
        return self.function(time_s, state)


@dataclass(frozen=True)
class _Loads:
    """Instantaneous acceleration/force diagnostics, with SI suffixes on physical fields."""

    acceleration_m_s2: FloatArray
    mass_flow_kg_s: float
    body_to_frame: FloatArray
    torque_body_n_m: FloatArray
    frame_rate_body_rad_s: FloatArray
    altitude_m: float
    relative_speed_m_s: float
    mach: float
    q_pa: float
    alpha_rad: float
    thrust_n: float
    axial_g: float
    twr: float


class _Dynamics:
    """Immutable geometry/configuration context and stage-specific force evaluation."""

    def __init__(self, vehicle: VehicleConfig, config: SimulationConfig,
                 direction: DirectionFunction | None, control: ControlFunction | None) -> None:
        """Cache launch geometry [m], frame bases, and angular velocity [rad/s]."""
        self.vehicle: VehicleConfig = vehicle
        self.config: SimulationConfig = config
        self.direction: DirectionFunction | None = direction
        self.control: ControlFunction | None = control
        self.origin_m: FloatArray = geodetic_to_ecef(config.latitude_rad, config.longitude_rad, config.launch_altitude_m)
        self.basis: FloatArray = enu_basis(config.latitude_rad, config.longitude_rad)
        self.omega_rad_s: FloatArray = np.array([0., 0., EARTH_OMEGA_RAD_S if config.rotating_earth else 0.])
        self.flight_session: FlightSession | None = None
        self.flight_command: FlightCommand | None = None
        self.rail_attached: bool = False
        self.rail_axis_ecef: FloatArray = np.zeros(3)

    def navigation(self, time_s: float, state: FloatArray, mode: _Mode) -> NavigationState:
        """Build a truth-navigation sample in ECI with physical fields in SI units."""
        position_m, velocity_m_s, frame_to_ecef = self.coordinates(time_s, state)
        ecef_to_eci = rotation_z(self.omega_rad_s[2]*time_s)
        latitude_rad, longitude_rad, altitude_m = ecef_to_geodetic(position_m)
        local_basis = enu_basis(float(latitude_rad), float(longitude_rad))
        wind_m_s = local_basis @ self.config.wind.velocity(float(altitude_m))
        stage = self.vehicle.stages[mode.stage_index]
        pressure_pa = float(atmosphere(float(altitude_m)).pressure_pa) if self.config.atmosphere_enabled else 0.
        engine = propulsion(stage, time_s-mode.ignition_s, pressure_pa, engine_on=mode.phase == "powered")
        return NavigationState(time_s, ecef_to_eci @ position_m,
                               ecef_to_eci @ (velocity_m_s+np.cross(self.omega_rad_s, position_m)),
                               ecef_to_eci @ (velocity_m_s-wind_m_s), ecef_to_eci @ local_basis,
                               ecef_to_eci @ self.basis,
                               ecef_to_eci @ frame_to_ecef @ quaternion_matrix(state[8:12]) if self.config.dof == 6 else np.eye(3),
                               state[12:15].copy() if self.config.dof == 6 else np.zeros(3),
                               state[15:17].copy() if len(state) == 17 else np.zeros(2),
                               float(state[6]), float(state[7]), float(altitude_m), float(engine.thrust_n),
                               float(engine.mass_flow_kg_s), state[6]*np.asarray(stage.principal_inertia_per_mass_m2),
                               mode.stage_index, time_s-mode.ignition_s, mode.phase == "powered", self.rail_attached, self.config.mu_m3_s2)

    def rail_distance_m(self, time_s: float, state: FloatArray) -> float:
        """Return along-rail distance [m] in the Earth-fixed launch geometry."""
        position_m, _, _ = self.coordinates(time_s, state)
        return float(np.dot(position_m-self.origin_m, self.rail_axis_ecef))

    def coordinates(self, time_s: float, state: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return Earth-fixed position [m], relative velocity [m/s], frame-to-ECEF basis."""
        if self.config.frame == "ECI":
            frame_to_ecef = rotation_z(-self.omega_rad_s[2] * time_s)
            position_m = frame_to_ecef @ state[:3]
            velocity_m_s = frame_to_ecef @ state[3:6] - np.cross(self.omega_rad_s, position_m)
        else:
            frame_to_ecef = self.basis
            position_m = self.origin_m + self.basis @ state[:3]
            velocity_m_s = self.basis @ state[3:6]
        return position_m, velocity_m_s, frame_to_ecef

    def altitude_m(self, time_s: float, state: FloatArray) -> float:
        """Return geodetic altitude [m] from current frame state."""
        position_m, _, _ = self.coordinates(time_s, state)
        return float(ecef_to_geodetic(position_m)[2])

    def loads(self, time_s: float, state: FloatArray, mode: _Mode) -> _Loads:
        """Compute SI environmental forces, rotating-frame terms, and body moments."""
        stage = self.vehicle.stages[mode.stage_index]
        position_m, velocity_m_s, frame_to_ecef = self.coordinates(time_s, state)
        latitude_rad, longitude_rad, altitude_m = ecef_to_geodetic(position_m)
        local_basis = enu_basis(float(latitude_rad), float(longitude_rad))
        if self.config.dof == 6:
            body_to_frame = quaternion_matrix(state[8:12])
            axis = body_to_frame[:, 0]
        else:
            axis = (frame_to_ecef.T @ local_basis[:, 2] if self.direction is None
                    else np.asarray(self.direction(time_s, state.copy()), dtype=float))
            if self.flight_command is not None:
                command_eci = self.flight_command.direction_eci(self.navigation(time_s, state, mode))
                axis = frame_to_ecef.T @ rotation_z(-self.omega_rad_s[2]*time_s) @ command_eci
            if axis.shape != (3,) or not np.all(np.isfinite(axis)) or not np.isclose(np.linalg.norm(axis), 1., atol=1e-8):
                raise ValueError("3DoF direction callback must return a finite unit vector in the propagation frame")
            body_to_frame = np.eye(3)  # unused rotation in 3DoF
        atmosphere_state = atmosphere(float(altitude_m))
        pressure_pa = float(atmosphere_state.pressure_pa) if self.config.atmosphere_enabled else 0.
        density_kg_m3 = float(atmosphere_state.density_kg_m3) if self.config.atmosphere_enabled else 0.
        wind_ecef_m_s = local_basis @ self.config.wind.velocity(float(altitude_m))
        relative_velocity_m_s = frame_to_ecef.T @ (velocity_m_s - wind_ecef_m_s)
        aero_n, mach, q_pa, alpha_rad = aerodynamic_force(
            relative_velocity_m_s, axis, density_kg_m3, float(atmosphere_state.speed_of_sound_m_s), stage.reference_area_m2, stage.aero)
        engine = propulsion(stage, time_s - mode.ignition_s, pressure_pa, engine_on=mode.phase == "powered")
        thrust_n, flow_kg_s = float(engine.thrust_n), float(engine.mass_flow_kg_s)
        control = _DEFAULT_CONTROL if self.control is None else self.control(time_s, state.copy())
        if not isinstance(control, ControlInput):
            raise TypeError("6DoF control callback must return ControlInput")
        thrust_direction_body = (gimbal_direction(state[15:17]) if self.flight_session is not None and self.config.dof == 6
                                 else np.asarray(control.thrust_direction_body))
        thrust_axis = axis if self.config.dof == 3 else body_to_frame @ thrust_direction_body
        force_n = thrust_n * thrust_axis + aero_n
        gravity_ecef_m_s2 = gravity_acceleration(position_m, mu_m3_s2=self.config.mu_m3_s2,
                                                j2=EARTH_J2 if self.config.j2_enabled else 0.)
        acceleration_m_s2 = frame_to_ecef.T @ gravity_ecef_m_s2 + force_n / state[6]
        if self.config.frame == "ENU":
            # Position is geocentric here, not the local ENU offset.
            acceleration_m_s2 += frame_to_ecef.T @ (
                -2. * np.cross(self.omega_rad_s, velocity_m_s)
                - np.cross(self.omega_rad_s, np.cross(self.omega_rad_s, position_m)))
        if self.rail_attached:
            rotation_terms_m_s2 = (2.*np.cross(self.omega_rad_s, velocity_m_s)
                                    + np.cross(self.omega_rad_s, np.cross(self.omega_rad_s, position_m)))
            rotating_acceleration_m_s2 = frame_to_ecef @ acceleration_m_s2
            if self.config.frame == "ECI":
                rotating_acceleration_m_s2 -= rotation_terms_m_s2
            constrained_m_s2 = self.rail_axis_ecef*np.dot(self.rail_axis_ecef, rotating_acceleration_m_s2)
            if self.config.frame == "ECI":
                constrained_m_s2 += rotation_terms_m_s2
            acceleration_m_s2 = frame_to_ecef.T @ constrained_m_s2
        torque_n_m = np.zeros(3)
        if self.config.dof == 6:
            torque_n_m = (np.asarray(control.torque_body_n_m)
                          + np.cross(stage.cp_offset_body_m, body_to_frame.T @ aero_n)
                          + np.cross(stage.thrust_offset_body_m, thrust_n * thrust_direction_body)
                          - stage.angular_damping_n_m_s * state[12:15])
        frame_rate_rad_s = (body_to_frame.T @ frame_to_ecef.T @ self.omega_rad_s
                            if self.config.frame == "ENU" else np.zeros(3))
        gravity_m_s2 = float(np.linalg.norm(gravity_ecef_m_s2))
        return _Loads(acceleration_m_s2, flow_kg_s, body_to_frame, torque_n_m, frame_rate_rad_s,
                      float(altitude_m), float(np.linalg.norm(relative_velocity_m_s)), mach, q_pa, alpha_rad,
                      thrust_n, float(np.dot(force_n, axis) / (state[6] * G0_M_S2)),
                      thrust_n / (state[6] * gravity_m_s2) if gravity_m_s2 > 0. else np.nan)

    def rhs(self, time_s: float, state: FloatArray, mode: _Mode) -> FloatArray:
        """Return SI state derivatives for one smooth event-bounded segment."""
        loads = self.loads(time_s, state, mode)
        derivative = np.zeros_like(state)
        derivative[:3], derivative[3:6] = state[3:6], loads.acceleration_m_s2
        derivative[6:8] = -loads.mass_flow_kg_s
        if self.config.dof == 6:
            rate_rad_s = state[12:15]
            derivative[8:12] = quaternion_derivative(state[8:12], rate_rad_s - loads.frame_rate_body_rad_s)
            inertia_per_mass_m2 = np.asarray(self.vehicle.stages[mode.stage_index].principal_inertia_per_mass_m2)
            inertia_kg_m2 = state[6] * inertia_per_mass_m2
            inertia_rate_kg_m2_s = -loads.mass_flow_kg_s * inertia_per_mass_m2
            derivative[12:15] = (loads.torque_body_n_m - np.cross(rate_rad_s, inertia_kg_m2 * rate_rad_s)
                                 - inertia_rate_kg_m2_s * rate_rad_s) / inertia_kg_m2
            if self.rail_attached:
                _, _, frame_to_ecef = self.coordinates(time_s, state)
                rail_rate_body_rad_s = loads.body_to_frame.T @ frame_to_ecef.T @ self.omega_rad_s
                derivative[8:12] = quaternion_derivative(state[8:12], rail_rate_body_rad_s-loads.frame_rate_body_rad_s)
                derivative[12:15] = 0.
            if self.flight_session is not None and self.flight_command is not None:
                derivative[15:17] = np.clip((self.flight_command.gimbal_target_rad-state[15:17])/self.flight_session.actuator_time_constant_s,
                                            -self.flight_session.actuator_rate_limit_rad_s, self.flight_session.actuator_rate_limit_rad_s)
        return derivative


def initial_state(vehicle: VehicleConfig, config: SimulationConfig) -> FloatArray:
    """Construct a fully fueled launch state in the configured propagation frame.

    Parameters
    ----------
    vehicle : VehicleConfig
        Stage, payload, and fairing masses [kg].
    config : SimulationConfig
        Launch position [m, rad], velocity [m/s], orientation, and rate [rad/s].

    Returns
    -------
    ndarray, shape (8,) or (15,)
        State with the SI column schema in the module docstring.
    """
    basis = enu_basis(config.latitude_rad, config.longitude_rad)
    origin_m = geodetic_to_ecef(config.latitude_rad, config.longitude_rad, config.launch_altitude_m)
    omega_rad_s = np.array([0., 0., EARTH_OMEGA_RAD_S if config.rotating_earth else 0.])
    state = np.zeros(8 if config.dof == 3 else 15)
    if config.frame == "ECI":
        state[:3] = origin_m
        state[3:6] = basis @ config.initial_velocity_enu_m_s + np.cross(omega_rad_s, origin_m)
    else:
        state[3:6] = config.initial_velocity_enu_m_s
    state[6:8] = vehicle.initial_mass_kg, vehicle.stages[0].propellant_mass_kg
    if config.dof == 6:
        axis_enu = np.array([np.cos(config.initial_elevation_rad) * np.sin(config.initial_azimuth_rad),
                             np.cos(config.initial_elevation_rad) * np.cos(config.initial_azimuth_rad),
                             np.sin(config.initial_elevation_rad)])
        # Choose the same physical roll reference before converting frames.
        reference = np.array([0., 0., 1.]) if abs(axis_enu[2]) < .9 else np.array([0., 1., 0.])
        side = np.cross(reference, axis_enu)
        side /= np.linalg.norm(side)
        body_to_enu = np.column_stack((axis_enu, side, np.cross(axis_enu, side)))
        body_to_frame = basis @ body_to_enu if config.frame == "ECI" else body_to_enu
        quaternion_xyzw = Rotation.from_matrix(body_to_frame).as_quat()
        state[8:12] = quaternion_xyzw[[3, 0, 1, 2]]
        state[12:15] = config.initial_body_rate_rad_s
    return state


def orbital_elements(position_eci_m: ArrayLike, velocity_eci_m_s: ArrayLike,
                     mu_m3_s2: float = EARTH_MU_M3_S2) -> dict[str, FloatArray]:
    """Return vectorized two-body osculating elements from ECI states.

    Parameters
    ----------
    position_eci_m, velocity_eci_m_s : array_like, shape (..., 3)
        Earth-centered inertial positions [m] and velocities [m/s].
    mu_m3_s2 : float
        Gravitational parameter [m^3/s^2]. Zero returns undefined (NaN) elements.

    Returns
    -------
    dict
        Semi-major axis [m], eccentricity (dimensionless), inclination [rad],
        apogee/perigee altitude [m] relative to the WGS84 equatorial radius,
        and two-body specific energy [m^2/s^2]. Unbound apogee is +inf;
        radial inclination is NaN. J2 or thrust makes these instantaneous only.
    """
    position_m, velocity_m_s = np.broadcast_arrays(np.asarray(position_eci_m, dtype=float), np.asarray(velocity_eci_m_s, dtype=float))
    if position_m.shape[-1:] != (3,) or not np.all(np.isfinite(position_m)) or not np.all(np.isfinite(velocity_m_s)):
        raise ValueError("Orbital states must be finite with final dimension three")
    radius_m = np.linalg.norm(position_m, axis=-1)
    if np.any(radius_m <= 0.) or not np.isfinite(mu_m3_s2) or mu_m3_s2 < 0.:
        raise ValueError("Require nonzero radius and finite nonnegative gravitational parameter")
    keys = ("semi_major_axis_m", "eccentricity", "inclination_rad", "apogee_altitude_m", "perigee_altitude_m", "specific_energy_m2_s2")
    if mu_m3_s2 == 0.:
        return {key: np.full_like(radius_m, np.nan) for key in keys}
    momentum_m2_s = np.cross(position_m, velocity_m_s)
    momentum_norm_m2_s = np.linalg.norm(momentum_m2_s, axis=-1)
    eccentricity_vector = np.cross(velocity_m_s, momentum_m2_s) / mu_m3_s2 - position_m / radius_m[..., None]
    eccentricity = np.linalg.norm(eccentricity_vector, axis=-1)
    energy_m2_s2 = .5 * np.sum(velocity_m_s**2, axis=-1) - mu_m3_s2 / radius_m
    parabolic = np.abs(energy_m2_s2) <= 1e-12 * mu_m3_s2 / radius_m
    semi_major_m = np.divide(-mu_m3_s2, 2. * energy_m2_s2, out=np.full_like(radius_m, np.inf), where=~parabolic)
    inclination_rad = np.arccos(np.clip(np.divide(momentum_m2_s[..., 2], momentum_norm_m2_s,
                                                 out=np.full_like(radius_m, np.nan), where=momentum_norm_m2_s > 1e-12), -1., 1.))
    perigee_m = momentum_norm_m2_s**2 / (mu_m3_s2 * (1. + eccentricity)) - WGS84_A_M
    apogee_m = np.where((energy_m2_s2 < 0.) & ~parabolic, semi_major_m * (1. + eccentricity) - WGS84_A_M, np.inf)
    return dict(zip(keys, (semi_major_m, eccentricity, inclination_rad, apogee_m, perigee_m, energy_m2_s2)))


def _events(dynamics: _Dynamics, mode: _Mode, fairing_attached: bool) -> list[_TerminalEvent]:
    """Build terminal residuals: impact/altitude [m], depletion [kg], scheduled time [s]."""
    config, vehicle = dynamics.config, dynamics.vehicle
    stage = vehicle.stages[mode.stage_index]
    events = [_TerminalEvent("ground_impact", lambda time_s, state: dynamics.altitude_m(time_s, state) - config.ground_altitude_m, -1.)]
    if mode.phase == "powered":
        events.extend([_TerminalEvent("propellant_depletion", lambda time_s, state: float(state[7]), -1.),
                       _TerminalEvent("scheduled_cutoff", lambda time_s, state: time_s - (mode.ignition_s + stage.nominal_cutoff_s))])
        if dynamics.flight_command is not None and np.isfinite(dynamics.flight_command.cutoff_time_s):
            events.append(_TerminalEvent("guided_cutoff", lambda time_s, state: time_s-dynamics.flight_command.cutoff_time_s))
    elif mode.phase in ("await_separation", "await_ignition"):
        events.append(_TerminalEvent(mode.phase, lambda time_s, state: time_s - mode.next_time_s))
    if fairing_attached and vehicle.fairing_jettison_time_s is not None:
        events.append(_TerminalEvent("fairing_jettison", lambda time_s, state: time_s - vehicle.fairing_jettison_time_s))
    elif fairing_attached and vehicle.fairing_jettison_altitude_m is not None:
        events.append(_TerminalEvent("fairing_jettison", lambda time_s, state: dynamics.altitude_m(time_s, state) - vehicle.fairing_jettison_altitude_m))
    if dynamics.rail_attached and dynamics.flight_session is not None:
        events.append(_TerminalEvent("rail_release", lambda time_s, state: dynamics.rail_distance_m(time_s, state)-dynamics.flight_session.rail_length_m))
    if dynamics.flight_command is not None and np.isfinite(dynamics.flight_command.next_update_s):
        events.append(_TerminalEvent("flight_update", lambda time_s, state: time_s-dynamics.flight_command.next_update_s))
    return events


def _transition(trigger: str, time_s: float, state: FloatArray, mode: _Mode,
                vehicle: VehicleConfig, fairing_attached: bool) -> tuple[str, FloatArray, _Mode, bool]:
    """Apply one event-driven discrete transition with no impulses in position/velocity."""
    state = state.copy()
    stage = vehicle.stages[mode.stage_index]
    name = trigger
    if trigger in ("propellant_depletion", "scheduled_cutoff", "guided_cutoff"):
        name = "MECO" if mode.stage_index == 0 else ("SECO" if mode.stage_index == 1 else f"stage_{mode.stage_index + 1}_cutoff")
        if trigger == "propellant_depletion":
            state[6] -= state[7]  # remove only root-roundoff residue, not step-based clipping
            state[7] = 0.
        mode = (_Mode("await_separation", mode.stage_index, mode.ignition_s, time_s + stage.separation_delay_s)
                if mode.stage_index + 1 < len(vehicle.stages) else _Mode("coast", mode.stage_index, mode.ignition_s))
    elif trigger == "await_separation":
        name = f"stage_{mode.stage_index + 1}_separation"
        state[6] -= stage.dry_mass_kg + state[7]
        state[7] = vehicle.stages[mode.stage_index + 1].propellant_mass_kg
        mode = _Mode("await_ignition", mode.stage_index + 1, time_s + stage.coast_after_separation_s,
                     time_s + stage.coast_after_separation_s)
    elif trigger == "await_ignition":
        name = "SES-1" if mode.stage_index == 1 else f"stage_{mode.stage_index + 1}_ignition"
        mode = _Mode("powered", mode.stage_index, time_s)
    elif trigger == "fairing_jettison":
        state[6] -= vehicle.fairing_mass_kg
        fairing_attached = False
    return name, state, mode, fairing_attached


def simulate(vehicle: VehicleConfig, config: SimulationConfig = SimulationConfig(), *,
             state0: ArrayLike | None = None, thrust_direction: DirectionFunction | None = None,
             control: ControlFunction | None = None, guidance: FlightProgram | None = None) -> SimulationResult:
    """Propagate one vehicle with terminal-event stitching and derived telemetry.

    Parameters
    ----------
    vehicle, config : VehicleConfig, SimulationConfig
        Validated SI vehicle and mission settings.
    state0 : array_like or None
        Optional state with the module schema, at elapsed time zero. Initial mass
        and propellant must match the fully fueled vehicle. None builds a launch state.
    thrust_direction : callable or None
        3DoF-only ``(time_s, state) -> unit axis`` in the propagation frame.
        Default is local geodetic up; this is not an orbital guidance program.
    control : callable or None
        6DoF-only ``(time_s, state) -> ControlInput``. Default is axial thrust,
        zero applied control torque. Callbacks must be pure and deterministic.
    guidance : FlightProgram or None
        Reusable sampled flight-computer factory. With dof=3, track its pure
        direction law ideally. With dof=6, append actual pitch/yaw gimbal angles
        [rad] as state columns 15:17 and integrate physical TVC servo dynamics.
        Cannot be combined with legacy direction/control callbacks.

    Returns
    -------
    SimulationResult
        Adaptive samples, event ledger with pre/post states, and telemetry.

    Raises
    ------
    ValueError
        For incompatible state/configuration or control inputs.
    RuntimeError
        If integration fails; failed partial trajectories are not reported as success.

    Notes
    -----
    Coincident events are re-rooted at the same epoch on successive segments,
    with priority impact, depletion/cutoff, stage transition, then fairing. The
    public history has one right-continuous row per epoch; the ledger preserves
    both sides of jumps. Initial ascent does not trigger the descending impact
    event; an unsupported launch can terminate at t=0. The last stage stays attached.
    """
    if (config.dof == 3 and control is not None) or (config.dof == 6 and thrust_direction is not None):
        raise ValueError("Use thrust_direction for 3DoF and control for 6DoF")
    if guidance is not None and (control is not None or thrust_direction is not None):
        raise ValueError("Guidance owns direction/control; do not also supply legacy callbacks")
    dynamics = _Dynamics(vehicle, config, thrust_direction, control)
    if guidance is not None:
        dynamics.flight_session = guidance.create_session(vehicle, config.dof)
        dynamics.rail_attached = dynamics.flight_session.rail_length_m > 0.
        dynamics.rail_axis_ecef = dynamics.basis @ dynamics.flight_session.rail_axis_enu
    state = initial_state(vehicle, config) if state0 is None else np.asarray(state0, dtype=float).copy()
    if dynamics.rail_attached and config.dof == 6 and state0 is None:
        rail = dynamics.flight_session.rail_axis_enu
        state = initial_state(vehicle, replace(config, initial_azimuth_rad=float(np.arctan2(rail[0], rail[1])),
                                               initial_elevation_rad=float(np.arcsin(rail[2]))))
        frame_to_ecef = np.eye(3) if config.frame == "ECI" else dynamics.basis
        state[12:15] = quaternion_matrix(state[8:12]).T @ frame_to_ecef.T @ dynamics.omega_rad_s
    if guidance is not None and config.dof == 6 and state.shape == (15,):
        state = np.concatenate((state, np.zeros(2)))
    if state.shape != (8 if config.dof == 3 else 17 if guidance is not None else 15,) or not np.all(np.isfinite(state)):
        raise ValueError("Initial state has wrong shape or nonfinite values")
    if not np.allclose(state[6:8], [vehicle.initial_mass_kg, vehicle.stages[0].propellant_mass_kg], rtol=0., atol=config.mass_atol_kg):
        raise ValueError("Initial total mass and active fuel must match the fully fueled vehicle")
    if dynamics.altitude_m(0., state) < config.ground_altitude_m - 1e-7:
        raise ValueError("Initial state is below the configured impact surface")
    if config.dof == 6 and not np.isclose(np.linalg.norm(state[8:12]), 1., atol=1e-10):
        raise ValueError("Initial quaternion must have unit norm")
    if dynamics.flight_session is not None and config.dof == 6 and np.any(np.abs(state[15:17]) > dynamics.flight_session.actuator_angle_limit_rad):
        raise ValueError("Initial gimbal angles exceed mechanical bounds")
    if dynamics.rail_attached:
        position_m, velocity_m_s, frame_to_ecef = dynamics.coordinates(0., state)
        lateral_m_s = velocity_m_s-dynamics.rail_axis_ecef*np.dot(velocity_m_s, dynamics.rail_axis_ecef)
        if np.linalg.norm(position_m-dynamics.origin_m) > 1e-6 or np.linalg.norm(lateral_m_s) > 1e-6:
            raise ValueError("Rail launch must start at the launch origin with along-rail velocity")
        if config.dof == 6:
            rotation = frame_to_ecef @ quaternion_matrix(state[8:12])
            if np.linalg.norm(rotation[:, 0]-dynamics.rail_axis_ecef) > 1e-6 or np.linalg.norm(state[12:15]-rotation.T @ dynamics.omega_rad_s) > 1e-6:
                raise ValueError("Initial attitude and body rate must match the Earth-fixed rail")
    if vehicle.fairing_jettison_altitude_m is not None and dynamics.altitude_m(0., state) > vehicle.fairing_jettison_altitude_m + 1e-7:
        raise ValueError("Initial state is already above the fairing trigger; use time=0 instead")
    mode = _Mode("powered" if config.engines_enabled else "coast")
    if dynamics.flight_session is not None:
        dynamics.flight_command = dynamics.flight_session.update(dynamics.navigation(0., state, mode))
    fairing_attached = vehicle.fairing_mass_kg > 0.
    time_s = 0.
    times_s: list[float] = [0.]
    states: list[FloatArray] = [state.copy()]
    modes: list[_Mode] = [mode]
    commands: list[FlightCommand | None] = [dynamics.flight_command]
    records: list[EventRecord] = []
    function_evaluations = 0
    termination = "duration"
    atol = np.array([config.position_atol_m] * 3 + [config.velocity_atol_m_s] * 3 + [config.mass_atol_kg] * 2
                    + ([config.quaternion_atol] * 4 + [config.rate_atol_rad_s] * 3 if config.dof == 6 else [])
                    + ([1e-9]*2 if guidance is not None and config.dof == 6 else []))
    while time_s < config.duration_s:
        events = _events(dynamics, mode, fairing_attached)
        # Bound trial evaluations by known clock events; the terminal residuals
        # still locate and dispatch every transition, including zero-delay events.
        clock_times_s = [config.duration_s]
        if mode.phase == "powered":
            clock_times_s.append(mode.ignition_s + vehicle.stages[mode.stage_index].nominal_cutoff_s)
        elif mode.phase in ("await_separation", "await_ignition"):
            clock_times_s.append(mode.next_time_s)
        if fairing_attached and vehicle.fairing_jettison_time_s is not None:
            clock_times_s.append(vehicle.fairing_jettison_time_s)
        if dynamics.flight_command is not None:
            clock_times_s.append(dynamics.flight_command.next_update_s)
            if mode.phase == "powered":
                clock_times_s.append(dynamics.flight_command.cutoff_time_s)
        segment_end_s = min(epoch_s for epoch_s in clock_times_s if epoch_s > time_s)
        solution = solve_ivp(lambda epoch_s, values: dynamics.rhs(epoch_s, values, mode),
                             (time_s, segment_end_s), state, method=config.method,
                             rtol=config.rtol, atol=atol, max_step=config.max_step_s,
                             first_step=min(.1 if guidance is not None and time_s > 0. else .01,
                                            config.max_step_s, segment_end_s - time_s), events=events)
        function_evaluations += solution.nfev
        if not solution.success:
            raise RuntimeError(f"Integration failed at {solution.t[-1]} s: {solution.message}")
        # Replace the shared endpoint so state and telemetry use the post-event mode.
        states[-1], modes[-1] = state.copy(), mode
        commands[-1] = dynamics.flight_command
        for index in range(1, len(solution.t)):
            time_resolution_s = 32.*np.finfo(float).eps*max(1., abs(float(solution.t[index])))
            if solution.t[index]-times_s[-1] > time_resolution_s:
                times_s.append(float(solution.t[index]))
                states.append(solution.y[:, index].copy())
                modes.append(mode)
                commands.append(dynamics.flight_command)
            elif solution.t[index] >= times_s[-1]:
                # Coalesce roundoff-separated event endpoints, retaining the
                # right-hand sample. The event ledger still retains exact roots.
                times_s[-1] = float(solution.t[index])
                states[-1], modes[-1] = solution.y[:, index].copy(), mode
                commands[-1] = dynamics.flight_command
        time_s, state = float(solution.t[-1]), solution.y[:, -1].copy()
        triggered = [index for index, roots_s in enumerate(solution.t_events) if len(roots_s)]
        if not triggered:
            break
        trigger = events[triggered[0]].name
        before, previous_stage = state.copy(), mode.stage_index
        name, state, mode, fairing_attached = _transition(trigger, time_s, state, mode, vehicle, fairing_attached)
        if trigger != "flight_update":
            records.append(EventRecord(name, trigger, time_s, previous_stage, before, state.copy()))
        states[-1], modes[-1] = state.copy(), mode
        if trigger == "ground_impact":
            termination = "ground_impact"
            break
        if trigger == "rail_release":
            dynamics.rail_attached = False
        if dynamics.flight_session is not None:
            dynamics.flight_command = dynamics.flight_session.update(dynamics.navigation(time_s, state, mode), trigger)
            commands[-1] = dynamics.flight_command
        if len(records) > 4 * len(vehicle.stages) + 8:
            raise RuntimeError("Unexpected repeated terminal event; refusing a zero-time restart loop")
    time_array_s, state_array = np.array(times_s), np.stack(states)
    telemetry = _telemetry(dynamics, time_array_s, state_array, modes, commands)
    return SimulationResult(time_array_s, state_array, np.array([item.stage_index for item in modes], dtype=np.int64),
                            np.array([item.phase == "powered" for item in modes], dtype=bool), tuple(records),
                            telemetry, config, termination, function_evaluations,
                            tuple(dynamics.flight_session.records) if dynamics.flight_session is not None else ())


def _telemetry(dynamics: _Dynamics, times_s: FloatArray, states: FloatArray, modes: list[_Mode],
               commands: list[FlightCommand | None]) -> dict[str, FloatArray]:
    """Build force diagnostics and vectorized inertial orbital telemetry with SI keys."""
    rows: list[_Loads] = []
    commanded_axes: list[FloatArray] = []
    attitude_errors_rad: list[float] = []
    for time_s, state, mode, command in zip(times_s, states, modes, commands):
        dynamics.flight_command = command
        dynamics.rail_attached = command.rail_attached if command is not None else False
        rows.append(dynamics.loads(float(time_s), state, mode))
        if command is not None:
            navigation = dynamics.navigation(float(time_s), state, mode)
            commanded_axis = command.direction_eci(navigation)
            commanded_axes.append(commanded_axis)
            attitude_errors_rad.append(float(np.arccos(np.clip(np.dot(navigation.body_to_eci[:, 0], commanded_axis), -1., 1.))) if dynamics.config.dof == 6 else 0.)
    names = ("altitude_m", "relative_speed_m_s", "mach", "q_pa", "alpha_rad", "thrust_n", "axial_g", "twr")
    telemetry = {name: np.array([getattr(row, name) for row in rows]) for name in names}
    if dynamics.config.frame == "ECI":
        position_eci_m, velocity_eci_m_s = states[:, :3], states[:, 3:6]
    else:
        position_ecef_m = states[:, :3] @ dynamics.basis.T + dynamics.origin_m
        velocity_ecef_m_s = states[:, 3:6] @ dynamics.basis.T + np.cross(dynamics.omega_rad_s, position_ecef_m)
        angle_rad = times_s * dynamics.omega_rad_s[2]
        cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
        position_eci_m, velocity_eci_m_s = position_ecef_m.copy(), velocity_ecef_m_s.copy()
        for source, destination in ((position_ecef_m, position_eci_m), (velocity_ecef_m_s, velocity_eci_m_s)):
            destination[:, 0] = cosine * source[:, 0] - sine * source[:, 1]
            destination[:, 1] = sine * source[:, 0] + cosine * source[:, 1]
    telemetry.update(orbital_elements(position_eci_m, velocity_eci_m_s, dynamics.config.mu_m3_s2))
    telemetry["position_eci_m"], telemetry["velocity_eci_m_s"] = position_eci_m, velocity_eci_m_s
    telemetry["mass_kg"], telemetry["propellant_kg"] = states[:, 6], states[:, 7]
    if dynamics.config.dof == 6:
        telemetry["quaternion_norm"] = np.linalg.norm(states[:, 8:12], axis=1)
    if commanded_axes:
        telemetry["command_axis_eci"] = np.stack(commanded_axes)
        telemetry["attitude_error_rad"] = np.array(attitude_errors_rad)
        telemetry["rail_attached"] = np.array([command.rail_attached for command in commands], dtype=float)
        if dynamics.config.dof == 6:
            telemetry["gimbal_rad"] = states[:, 15:17]
            telemetry["gimbal_target_rad"] = np.stack([command.gimbal_target_rad for command in commands])
    return telemetry
