"""Physics-owned guidance interfaces; the propagator never imports guidance code.

Notes
-----
Navigation uses ECI vectors irrespective of the propagation frame. Time is mission
elapsed seconds. Truth navigation is the default; sensor models can implement the
same interface later. Array axes and flags are dimensionless unless named otherwise.
"""

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from .vehicle import VehicleConfig

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class NavigationState:
    """Accepted or trial navigation sample, with SI units in physical field names.

    Notes
    -----
    Matrices map body/local ENU/launch ENU axes into ECI. ``body_rate_rad_s``
    is inertial angular velocity expressed in body axes. Stage index is zero-based.
    Gimbal order is pitch, yaw. ``rail_attached`` is an event-owned discrete flag.
    """

    time_s: float
    position_eci_m: FloatArray
    velocity_eci_m_s: FloatArray
    relative_velocity_eci_m_s: FloatArray
    local_to_eci: FloatArray
    launch_to_eci: FloatArray
    body_to_eci: FloatArray
    body_rate_rad_s: FloatArray
    gimbal_rad: FloatArray
    mass_kg: float
    propellant_kg: float
    altitude_m: float
    thrust_n: float
    mass_flow_kg_s: float
    inertia_kg_m2: FloatArray
    stage_index: int
    stage_elapsed_s: float
    engine_on: bool
    rail_attached: bool
    mu_m3_s2: float


@dataclass(frozen=True)
class FlightCommand:
    """Pure direction law and held actuator setpoint valid until an accepted update.

    Attributes
    ----------
    direction_eci : callable
        Pure navigation-to-unit-axis law (dimensionless); must not mutate memory.
    next_update_s, cutoff_time_s : float
        Absolute clock events [s]. Infinity disables a guided cutoff.
    gimbal_target_rad : ndarray, shape (2,)
        Held pitch/yaw actuator setpoint [rad].
    phase : str
        Human-readable mode, not a physical quantity.
    rail_attached : bool
        Rail state associated with this command/history sample.
    """

    direction_eci: Callable[[NavigationState], FloatArray]
    next_update_s: float
    cutoff_time_s: float = np.inf
    gimbal_target_rad: FloatArray = field(default_factory=lambda: np.zeros(2))
    phase: str = "coast"
    rail_attached: bool = False


class FlightSession(Protocol):
    """Per-run sampled flight computer contract with SI actuator/rail parameters."""

    rail_length_m: float
    rail_axis_enu: FloatArray
    actuator_time_constant_s: float
    actuator_rate_limit_rad_s: float
    actuator_angle_limit_rad: float
    records: list[dict[str, object]]

    def update(self, navigation: NavigationState, event: str | None = None) -> FlightCommand:
        """Update memory only at an accepted time/event and return an immutable command."""
        ...


class FlightProgram(Protocol):
    """Reusable program factory; separate simulations must receive separate memory."""

    def create_session(self, vehicle: VehicleConfig, dof: int) -> FlightSession:
        """Create a run-local session for vehicle SI inputs and dimensionless DoF selector."""
        ...


def gimbal_direction(gimbal_rad: FloatArray) -> FloatArray:
    """Map pitch/yaw gimbal angles [rad] to a body thrust unit vector.

    Returns
    -------
    ndarray, shape (3,)
        [cos(pitch)cos(yaw), cos(pitch)sin(yaw), -sin(pitch)], dimensionless.
    """
    pitch_rad, yaw_rad = gimbal_rad
    return np.array([np.cos(pitch_rad)*np.cos(yaw_rad), np.cos(pitch_rad)*np.sin(yaw_rad), -np.sin(pitch_rad)])
