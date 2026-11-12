"""Continuous engineering aerodynamic surrogate and altitude-sheared ENU winds.

Notes
-----
Coefficients are dimensionless, not a measured database. All vectors end in
three components. Wind components use local east, north, up axes, in m/s.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AeroConfig:
    """Dimensionless surrogate coefficients; replace with calibrated data.

    Attributes
    ----------
    base_cd, wave_cd, hypersonic_cd : float
        Subsonic drag, added transonic peak, and axial hypersonic drag limit.
    lift_slope_per_rad : float
        Small-angle lift slope [1/rad].
    """

    base_cd: float = .25
    wave_cd: float = .45
    hypersonic_cd: float = .15
    lift_slope_per_rad: float = 2.0

    def __post_init__(self) -> None:
        """Reject nonfinite or negative dimensionless aerodynamic parameters."""
        values = (self.base_cd, self.wave_cd, self.hypersonic_cd, self.lift_slope_per_rad)
        if not np.all(np.isfinite(values)) or min(values) < 0.:
            raise ValueError("Aerodynamic coefficients must be finite and nonnegative")


def coefficients(mach: ArrayLike, alpha_rad: ArrayLike, config: AeroConfig = AeroConfig()) -> tuple[FloatArray, FloatArray]:
    """Return continuous drag and signed lift coefficients (dimensionless).

    Parameters
    ----------
    mach : array_like
        Nonnegative Mach number (dimensionless).
    alpha_rad : array_like
        Signed angle of attack [rad], in [-pi, pi].
    config : AeroConfig
        Dimensionless coefficient parameters.

    Returns
    -------
    cd, cl : ndarray
        Broadcast coefficients; drag is even and lift is odd in alpha.

    Notes
    -----
    A smooth sigmoid raises wave drag over Mach 0.85--1.25, followed by a
    supersonic decay. At high Mach the incidence contribution approaches
    Newtonian sin^2(alpha) drag and signed sin^2(alpha)*cos(alpha) lift.
    """
    mach, alpha_rad = np.broadcast_arrays(np.asarray(mach, dtype=float), np.asarray(alpha_rad, dtype=float))
    if np.any(~np.isfinite(mach)) or np.any(mach < 0.) or np.any(~np.isfinite(alpha_rad)) or np.any(np.abs(alpha_rad) > np.pi):
        raise ValueError("Require finite Mach >= 0 and |alpha_rad| <= pi")
    wave_onset = .5 * (1. + np.tanh((mach - .95) / .08))
    wave_decay = 1. / (1. + (np.maximum(mach - 1.1, 0.) / .8) ** 2)
    hypersonic_weight = mach**4 / (mach**4 + 4.**4)
    cd_axial = (1. - hypersonic_weight) * config.base_cd + hypersonic_weight * config.hypersonic_cd
    lift_slope = ((1. - hypersonic_weight) * config.lift_slope_per_rad
                  + 2. * hypersonic_weight * np.abs(np.sin(alpha_rad)))
    cd = cd_axial + config.wave_cd * wave_onset * wave_decay + 2. * np.sin(alpha_rad)**2
    cl = lift_slope * np.sin(alpha_rad) * np.cos(alpha_rad)
    return np.asarray(cd), np.asarray(cl)


@dataclass(frozen=True)
class WindProfile:
    """Piecewise-linear wind shear with constant endpoint extrapolation.

    Parameters
    ----------
    altitude_m : tuple of float
        Strictly increasing geometric heights [m].
    velocity_enu_m_s : tuple of 3-tuples
        East/north/up wind velocity [m/s] at each height, relative to Earth.
    """

    altitude_m: tuple[float, ...] = (0.,)
    velocity_enu_m_s: tuple[tuple[float, float, float], ...] = ((0., 0., 0.),)

    def __post_init__(self) -> None:
        """Validate wind heights [m] and three-component velocities [m/s]."""
        heights_m = np.asarray(self.altitude_m)
        winds_m_s = np.asarray(self.velocity_enu_m_s)
        if (heights_m.ndim != 1 or not heights_m.size or winds_m_s.shape != (heights_m.size, 3)
                or not np.all(np.isfinite(heights_m)) or not np.all(np.isfinite(winds_m_s))
                or np.any(np.diff(heights_m) <= 0.)):
            raise ValueError("Wind profile requires increasing finite heights and (n, 3) velocities")

    def velocity(self, altitude_m: ArrayLike) -> FloatArray:
        """Interpolate wind shear in local geographic ENU axes.

        Parameters
        ----------
        altitude_m : array_like
            Finite geometric heights [m].

        Returns
        -------
        ndarray
            Wind [m/s], with input shape plus a final east/north/up axis.
        """
        altitude_m = np.asarray(altitude_m, dtype=float)
        if np.any(~np.isfinite(altitude_m)):
            raise ValueError("Wind altitude must be finite")
        winds_m_s = np.asarray(self.velocity_enu_m_s)
        return np.stack([np.interp(altitude_m, self.altitude_m, winds_m_s[:, axis]) for axis in range(3)], axis=-1)


def dynamic_pressure(density_kg_m3: ArrayLike, velocity_m_s: ArrayLike, wind_m_s: ArrayLike = (0., 0., 0.)) -> FloatArray:
    """Evaluate dynamic pressure from same-frame three-dimensional relative flow.

    Parameters
    ----------
    density_kg_m3 : array_like
        Nonnegative mass density [kg/m^3].
    velocity_m_s, wind_m_s : array_like, shape (..., 3)
        Vehicle and wind velocities [m/s] expressed in the same frame.

    Returns
    -------
    ndarray
        Dynamic pressure [Pa], broadcast over all leading dimensions.
    """
    density_kg_m3 = np.asarray(density_kg_m3, dtype=float)
    velocity_m_s, wind_m_s = np.asarray(velocity_m_s, dtype=float), np.asarray(wind_m_s, dtype=float)
    if velocity_m_s.shape[-1:] != (3,) or wind_m_s.shape[-1:] != (3,):
        raise ValueError("Velocity and wind must end in three components")
    if (np.any(density_kg_m3 < 0.) or not np.all(np.isfinite(density_kg_m3))
            or not np.all(np.isfinite(velocity_m_s)) or not np.all(np.isfinite(wind_m_s))):
        raise ValueError("Require finite velocities and nonnegative finite density")
    relative_m_s = velocity_m_s - wind_m_s
    return np.asarray(.5 * density_kg_m3 * np.sum(relative_m_s**2, axis=-1))


def aerodynamic_force(relative_velocity_m_s: ArrayLike, body_axis: ArrayLike, density_kg_m3: float,
                      sound_speed_m_s: float, reference_area_m2: float,
                      config: AeroConfig = AeroConfig()) -> tuple[FloatArray, float, float, float]:
    """Resolve drag and lift into one frame, including zero-speed degeneracy.

    Parameters
    ----------
    relative_velocity_m_s, body_axis : array_like, shape (3,)
        Air-relative velocity [m/s] and unit forward axis (dimensionless), same frame.
    density_kg_m3, sound_speed_m_s, reference_area_m2 : float
        Density [kg/m^3], sound speed [m/s], and coefficient reference area [m^2].
    config : AeroConfig
        Coefficient model.

    Returns
    -------
    force_n, mach, q_pa, alpha_rad : tuple
        Force vector [N], Mach (dimensionless), Q [Pa], total incidence [rad].
    """
    velocity_m_s = np.asarray(relative_velocity_m_s, dtype=float)
    axis = np.asarray(body_axis, dtype=float)
    if velocity_m_s.shape != (3,) or axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError("Force vectors must have shape (3,) and finite components")
    if not np.isclose(np.linalg.norm(axis), 1., atol=1e-8):
        raise ValueError("body_axis must be a unit vector")
    if not np.isfinite(sound_speed_m_s) or sound_speed_m_s <= 0. or not np.isfinite(reference_area_m2) or reference_area_m2 < 0.:
        raise ValueError("Sound speed must be positive and area nonnegative")
    q_pa = float(dynamic_pressure(density_kg_m3, velocity_m_s))
    speed_m_s = float(np.linalg.norm(velocity_m_s))
    if speed_m_s == 0.:
        return np.zeros(3), 0., q_pa, 0.
    velocity_axis = velocity_m_s / speed_m_s
    cos_alpha = float(np.clip(np.dot(axis, velocity_axis), -1., 1.))
    lift_axis = axis - cos_alpha * velocity_axis
    sin_alpha = float(np.linalg.norm(lift_axis))
    alpha_rad = float(np.arctan2(sin_alpha, cos_alpha))
    mach = speed_m_s / sound_speed_m_s
    cd, cl = coefficients(mach, alpha_rad, config)
    lift_axis = lift_axis / sin_alpha if sin_alpha > 1e-12 else np.zeros(3)
    force_n = q_pa * reference_area_m2 * (-float(cd) * velocity_axis + float(cl) * lift_axis)
    return force_n, mach, q_pa, alpha_rad
