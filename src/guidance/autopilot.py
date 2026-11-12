"""Quaternion-error PID/LQR control with physical two-axis TVC allocation.

Notes
-----
Body +x is the nose, nozzle offsets must lie on negative body x. A single
two-axis nozzle cannot control roll: no fictitious roll torque is supplied.
Controller memory changes only on sampled updates, never inside the ODE RHS.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_continuous_are
from scipy.spatial.transform import Rotation

from src.physics.frames import quaternion_derivative, quaternion_matrix

FloatArray = NDArray[np.float64]


def attitude_quaternion(direction: ArrayLike, up_reference: ArrayLike) -> FloatArray:
    """Construct a scalar-first body-to-frame quaternion from unit nose and up axes.

    Parameters
    ----------
    direction, up_reference : array_like, shape (3,)
        Desired nose and roll reference in one frame (dimensionless).

    Returns
    -------
    ndarray, shape (4,)
        Hamilton [w,x,y,z] quaternion (dimensionless). At vertical, choose a
        deterministic orthogonal roll reference; roll is not controlled by TVC.
    """
    nose = np.asarray(direction, dtype=float)
    up = np.asarray(up_reference, dtype=float)
    if nose.shape != (3,) or up.shape != (3,) or not np.all(np.isfinite([nose, up])) or np.linalg.norm(nose) < 1e-12:
        raise ValueError("Attitude directions must be finite three-vectors with a nonzero nose")
    nose = nose / np.linalg.norm(nose)
    side = np.cross(up, nose)
    if np.linalg.norm(side) < 1e-8:
        reference = np.eye(3)[int(np.argmin(np.abs(nose)))]
        side = np.cross(reference, nose)
    side /= np.linalg.norm(side)
    quaternion_xyzw = Rotation.from_matrix(np.column_stack((nose, side, np.cross(nose, side)))).as_quat()
    return quaternion_xyzw[[3, 0, 1, 2]]


def quaternion_error(current: ArrayLike, commanded: ArrayLike) -> FloatArray:
    """Return shortest-arc body-frame rotation error [rad] between Hamilton quaternions.

    Parameters
    ----------
    current, commanded : array_like, shape (4,)
        Scalar-first body-to-common-frame quaternions (dimensionless).

    Returns
    -------
    ndarray, shape (3,)
        Rotation vector [rad], invariant to quaternion sign.
    """
    relative = quaternion_matrix(current).T @ quaternion_matrix(commanded)
    return Rotation.from_matrix(relative).as_rotvec()


def attitude_kinematics(quaternion: ArrayLike, body_rate_rad_s: ArrayLike) -> FloatArray:
    """Return q_dot [1/s] = 0.5 q*(0,omega) for a body-to-inertial quaternion.

    Parameters
    ----------
    quaternion : array_like, shape (4,)
        Scalar-first quaternion (dimensionless).
    body_rate_rad_s : array_like, shape (3,)
        Body angular velocity relative to the quaternion frame [rad/s].
    """
    quaternion_matrix(quaternion)  # validate nonzero, finite orientation
    rate_rad_s = np.asarray(body_rate_rad_s, dtype=float)
    if rate_rad_s.shape != (3,) or not np.all(np.isfinite(rate_rad_s)):
        raise ValueError("Angular rate must be a finite three-vector")
    return quaternion_derivative(quaternion, rate_rad_s)


@dataclass(frozen=True)
class AutopilotConfig:
    """Sampled controller and actuator settings in SI units.

    Parameters
    ----------
    mode : {"PID", "LQR"}
        Controller selector (dimensionless).
    update_period_s, actuator_time_constant_s, command_rate_filter_s : float
        Control sample period, servo lag, and command-rate low-pass time [s].
    gimbal_limit_rad, gimbal_rate_limit_rad_s : float
        Per-axis mechanical travel [rad] and slew limit [rad/s].
    natural_frequency_rad_s, damping_ratio, integral_gain_per_s3 : float
        PID bandwidth [rad/s], dimensionless damping, integral gain [1/s^3].
    integral_limit_rad_s : float
        Per-axis attitude-error integral bound [rad*s].
    lqr_angle_weight, lqr_rate_weight, lqr_effort_weight : float
        Positive dimensionless costs on states normalized by 1 rad and 1 rad/s,
        and angular acceleration normalized by 1 rad/s^2.
    """

    mode: Literal["PID", "LQR"] = "PID"
    update_period_s: float = .1
    actuator_time_constant_s: float = .08
    command_rate_filter_s: float = .5
    gimbal_limit_rad: float = np.deg2rad(5.)
    gimbal_rate_limit_rad_s: float = np.deg2rad(20.)
    natural_frequency_rad_s: float = 1.2
    damping_ratio: float = .9
    integral_gain_per_s3: float = .08
    integral_limit_rad_s: float = .2
    lqr_angle_weight: float = 2.0736
    lqr_rate_weight: float = 1.7856
    lqr_effort_weight: float = 1.

    def __post_init__(self) -> None:
        """Validate positive settings and meaningful mechanical gimbal bounds [rad]."""
        positive = (self.update_period_s, self.actuator_time_constant_s, self.command_rate_filter_s, self.gimbal_limit_rad,
                    self.gimbal_rate_limit_rad_s, self.natural_frequency_rad_s, self.damping_ratio,
                    self.integral_limit_rad_s, self.lqr_angle_weight, self.lqr_rate_weight, self.lqr_effort_weight)
        if self.mode not in ("PID", "LQR") or not np.all(np.isfinite(positive)) or min(positive) <= 0.:
            raise ValueError("Controller selectors and positive gains/limits must be valid")
        if self.gimbal_limit_rad >= np.pi/4. or not np.isfinite(self.integral_gain_per_s3) or self.integral_gain_per_s3 < 0.:
            raise ValueError("Require gimbal limit below 45 degrees and nonnegative integral gain")


class TVCAutopilot:
    """Run-local PID/LQR attitude controller; initialize a new instance for each run."""

    def __init__(self, config: AutopilotConfig = AutopilotConfig()) -> None:
        """Initialize SI controller memory and the dimensionless normalized LQR design."""
        self.config: AutopilotConfig = config
        self.integral_rad_s: FloatArray = np.zeros(2)
        self.last_time_s: float | None = None
        self.last_direction: FloatArray | None = None
        self.command_rate_eci_rad_s: FloatArray = np.zeros(3)
        self.last_target_rad: FloatArray = np.zeros(2)
        self.saturated: bool = False
        self.kp_per_s2: float = config.natural_frequency_rad_s**2
        self.kd_per_s: float = 2.*config.damping_ratio*config.natural_frequency_rad_s
        if config.mode == "LQR":
            system = np.array([[0., 1.], [0., 0.]])
            actuator = np.array([[0.], [1.]])
            cost = np.diag([config.lqr_angle_weight, config.lqr_rate_weight])
            effort = np.array([[config.lqr_effort_weight]])
            riccati = solve_continuous_are(system, actuator, cost, effort)
            gains = (actuator.T @ riccati / config.lqr_effort_weight)[0]
            self.kp_per_s2, self.kd_per_s = float(gains[0]), float(gains[1])

    def reset(self) -> None:
        """Reset integrated error [rad*s] and sampled history after a stage/rail transition."""
        self.integral_rad_s[:] = 0.
        self.last_time_s, self.last_direction = None, None
        self.command_rate_eci_rad_s[:] = 0.
        self.last_target_rad[:] = 0.
        self.saturated = False

    def command(self, time_s: float, body_to_eci: FloatArray, body_rate_rad_s: FloatArray,
                direction_eci: FloatArray, inertia_kg_m2: FloatArray, inertia_rate_kg_m2_s: FloatArray,
                thrust_n: float, lever_arm_m: float, actual_gimbal_rad: FloatArray) -> FloatArray:
        """Update control and allocate pitch/yaw gimbal targets [rad].

        Parameters
        ----------
        time_s : float
            Accepted sample time [s], monotonically increasing (same-time events allowed).
        body_to_eci, direction_eci : ndarray
            Actual rotation and desired nose unit vector (dimensionless).
        body_rate_rad_s : ndarray, shape (3,)
            Actual inertial body rate [rad/s].
        inertia_kg_m2, inertia_rate_kg_m2_s : ndarray, shape (3,)
            Principal moments [kg*m^2] and derivatives [kg*m^2/s].
        thrust_n, lever_arm_m : float
            Available positive thrust [N] and distance to an aft nozzle [m].
        actual_gimbal_rad : ndarray, shape (2,)
            Measured actuator pitch/yaw [rad].

        Returns
        -------
        ndarray, shape (2,)
            Mechanically bounded pitch/yaw setpoint [rad]. Integral updates stop
            during angle/rate saturation or loss of thrust authority.
        """
        arrays = (body_to_eci, body_rate_rad_s, direction_eci, inertia_kg_m2, inertia_rate_kg_m2_s, actual_gimbal_rad)
        shapes = ((3, 3), (3,), (3,), (3,), (3,), (2,))
        if any(np.shape(array) != shape or not np.all(np.isfinite(array)) for array, shape in zip(arrays, shapes)):
            raise ValueError("Controller arrays must have documented shapes and finite SI values")
        if (not np.all(np.isfinite([time_s, thrust_n, lever_arm_m])) or np.any(inertia_kg_m2 <= 0.)
                or not np.isclose(np.linalg.norm(direction_eci), 1., atol=1e-8)):
            raise ValueError("Controller requires finite inputs, positive inertia, and unit desired direction")
        if self.last_time_s is not None and time_s < self.last_time_s - 1e-12:
            raise ValueError("Controller cannot update on a rejected/backward solver time")
        dt_s = 0. if self.last_time_s is None else time_s - self.last_time_s
        desired_body = body_to_eci.T @ direction_eci
        # Minimal nose alignment is a quaternion rotation with no demanded roll.
        axis = np.cross([1., 0., 0.], desired_body)
        sine = np.linalg.norm(axis)
        angle_rad = np.arctan2(sine, np.clip(desired_body[0], -1., 1.))
        error_rad = axis * angle_rad/sine if sine > 1e-12 else np.array([0., angle_rad, 0.])
        if self.last_direction is not None and dt_s > 1e-12:
            raw_rate_eci_rad_s = np.cross(self.last_direction, direction_eci) / dt_s
            weight = dt_s/(self.config.command_rate_filter_s+dt_s)
            self.command_rate_eci_rad_s += weight*(raw_rate_eci_rad_s-self.command_rate_eci_rad_s)
        rate_error_rad_s = body_rate_rad_s - body_to_eci.T @ self.command_rate_eci_rad_s
        if thrust_n <= 1e-6 or lever_arm_m <= 0.:
            self.reset()
            self.last_time_s = time_s
            self.last_direction = direction_eci.copy()
            return np.zeros(2)
        candidate_rad_s = self.integral_rad_s.copy()
        tracking_actuator = np.max(np.abs(actual_gimbal_rad - self.last_target_rad)) < np.deg2rad(.1)
        if self.config.mode == "PID" and not self.saturated and tracking_actuator:
            candidate_rad_s = np.clip(candidate_rad_s + error_rad[1:]*dt_s,
                                     -self.config.integral_limit_rad_s, self.config.integral_limit_rad_s)
        acceleration_rad_s2 = self.kp_per_s2*error_rad - self.kd_per_s*rate_error_rad_s
        if self.config.mode == "PID":
            acceleration_rad_s2[1:] += self.config.integral_gain_per_s3*candidate_rad_s
        torque_n_m = (inertia_kg_m2*acceleration_rad_s2
                      + np.cross(body_rate_rad_s, inertia_kg_m2*body_rate_rad_s)
                      + inertia_rate_kg_m2_s*body_rate_rad_s)
        pitch_rad = -np.arcsin(np.clip(torque_n_m[1]/(thrust_n*lever_arm_m), -1., 1.))
        yaw_rad = np.arcsin(np.clip(-torque_n_m[2]/(thrust_n*lever_arm_m*max(np.cos(pitch_rad), 1e-8)), -1., 1.))
        raw_rad = np.array([pitch_rad, yaw_rad])
        target_rad = np.clip(raw_rad, -self.config.gimbal_limit_rad, self.config.gimbal_limit_rad)
        self.saturated = bool(np.any(np.abs(raw_rad) > self.config.gimbal_limit_rad))
        if not self.saturated:
            self.integral_rad_s = candidate_rad_s
        self.last_time_s, self.last_direction = time_s, direction_eci.copy()
        self.last_target_rad = target_rad.copy()
        return target_rad
