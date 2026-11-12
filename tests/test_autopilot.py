"""Analytical quaternion, controller sign, saturation, and LQR stability checks."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.guidance import AutopilotConfig, TVCAutopilot, attitude_kinematics, attitude_quaternion, quaternion_error
from src.physics.flight import gimbal_direction
from src.physics.frames import quaternion_matrix


def test_quaternion_kinematics_sign_and_double_cover() -> None:
    """Identity rotation with omega [rad/s] must give q_dot [1/s] = [0,omega/2]."""
    rate_rad_s = np.array([.1, -.2, .3])
    assert_allclose(attitude_kinematics([1., 0., 0., 0.], rate_rad_s), [0., .05, -.1, .15])
    quaternion = attitude_quaternion([1., .1, .05], [0., 0., 1.])
    assert_allclose(quaternion_error(quaternion, -quaternion), 0., atol=1e-15)
    assert_allclose(quaternion_matrix(quaternion)[:, 0], np.array([1., .1, .05])/np.linalg.norm([1., .1, .05]))


@pytest.mark.parametrize("mode", ["PID", "LQR"])
def test_tvc_moment_sign_and_stable_linear_feedback(mode: str) -> None:
    """Nozzle moments must correct a positive yaw error; local closed-loop eigenvalues must be stable."""
    controller = TVCAutopilot(AutopilotConfig(mode=mode))
    target = np.array([np.cos(.01), np.sin(.01), 0.])
    angles_rad = controller.command(0., np.eye(3), np.zeros(3), target, np.ones(3)*10.,
                                    np.zeros(3), 100., 2., np.zeros(2))
    torque_n_m = np.cross([-2., 0., 0.], 100.*gimbal_direction(angles_rad))
    assert torque_n_m[2] > 0. and angles_rad[1] < 0.
    poles_per_s = np.linalg.eigvals([[0., 1.], [-controller.kp_per_s2, -controller.kd_per_s]])
    assert np.all(poles_per_s.real < 0.)


def test_saturation_antiwindup_and_no_thrust_authority() -> None:
    """Gimbal targets stay within +/-5 degrees and sustained saturation does not grow the integral."""
    controller = TVCAutopilot()
    for time_s in np.arange(0., 10., .1):
        command_rad = controller.command(float(time_s), np.eye(3), np.zeros(3), np.array([0., 1., 0.]),
                                           np.ones(3)*1000., np.zeros(3), 1., 1., np.zeros(2))
        assert np.max(np.abs(command_rad)) <= np.deg2rad(5.)
    assert controller.saturated
    assert_allclose(controller.integral_rad_s, 0.)
    off_rad = controller.command(10., np.eye(3), np.zeros(3), np.array([0., 1., 0.]),
                                  np.ones(3), np.zeros(3), 0., 1., np.zeros(2))
    assert_allclose(off_rad, 0.)
    assert_allclose(controller.integral_rad_s, 0.)


def test_controller_rejects_backwards_updates() -> None:
    """Rejected solver times cannot mutate sampled controller memory."""
    controller = TVCAutopilot()
    args = (np.eye(3), np.zeros(3), np.array([1., 0., 0.]), np.ones(3), np.zeros(3), 100., 1., np.zeros(2))
    controller.command(1., *args)
    with pytest.raises(ValueError):
        controller.command(.9, *args)


@pytest.mark.parametrize("changes", [{"mode": "bad"}, {"gimbal_limit_rad": 0.}, {"update_period_s": 0.},
                                     {"gimbal_rate_limit_rad_s": -1.}, {"integral_gain_per_s3": -1.}])
def test_invalid_autopilot_settings(changes: dict[str, object]) -> None:
    """Reject nonphysical control gains and actuator limits before integration."""
    with pytest.raises(ValueError):
        replace(AutopilotConfig(), **changes)
