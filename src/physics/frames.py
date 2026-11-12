"""WGS84 geometry, ECI/Earth-fixed transforms, and scalar-first quaternions.

Notes
-----
ECI and Earth-fixed axes coincide at mission elapsed time zero. Precession,
nutation, polar motion, and UT1/UTC conversion are outside this model. Quaternions
are active Hamilton rotations [w, x, y, z] mapping body vectors into a frame.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import EARTH_J2, EARTH_MU_M3_S2, WGS84_A_M, WGS84_F

FloatArray = NDArray[np.float64]


def geodetic_to_ecef(latitude_rad: float, longitude_rad: float, altitude_m: float) -> FloatArray:
    """Convert WGS84 geodetic coordinates to an Earth-fixed position.

    Parameters
    ----------
    latitude_rad, longitude_rad : float
        Geodetic latitude and longitude [rad].
    altitude_m : float
        Ellipsoid-normal height [m].

    Returns
    -------
    ndarray, shape (3,)
        Earth-centered, Earth-fixed position [m].
    """
    if not np.all(np.isfinite([latitude_rad, longitude_rad, altitude_m])) or abs(latitude_rad) > np.pi / 2.:
        raise ValueError("Invalid finite geodetic coordinates")
    eccentricity_sq = WGS84_F * (2. - WGS84_F)
    normal_m = WGS84_A_M / np.sqrt(1. - eccentricity_sq * np.sin(latitude_rad)**2)
    return np.array([(normal_m + altitude_m) * np.cos(latitude_rad) * np.cos(longitude_rad),
                     (normal_m + altitude_m) * np.cos(latitude_rad) * np.sin(longitude_rad),
                     (normal_m * (1. - eccentricity_sq) + altitude_m) * np.sin(latitude_rad)])


def ecef_to_geodetic(position_m: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return geodetic latitude [rad], longitude [rad], height [m] for (..., 3) positions.

    Parameters
    ----------
    position_m : array_like, shape (..., 3)
        Earth-centered position [m] in Earth-fixed axes.

    Returns
    -------
    latitude_rad, longitude_rad, altitude_m : ndarray
        WGS84 latitude [rad], longitude [rad], and ellipsoid-normal height [m].

    Notes
    -----
    Fixed-point latitude iteration is vectorized over positions. The height
    formula projects onto the surface normal and remains regular at the poles.
    Intended for the surface and exterior of Earth, not its deep interior.
    """
    position_m = np.asarray(position_m, dtype=float)
    if position_m.shape[-1:] != (3,) or not np.all(np.isfinite(position_m)):
        raise ValueError("Positions must be finite and end in three components")
    if np.any(np.linalg.norm(position_m, axis=-1) < .5 * WGS84_A_M):
        raise ValueError("Geodetic conversion is undefined in the deep interior")
    eccentricity_sq = WGS84_F * (2. - WGS84_F)
    horizontal_m = np.hypot(position_m[..., 0], position_m[..., 1])
    z_m = position_m[..., 2]
    latitude_rad = np.arctan2(z_m, horizontal_m * (1. - eccentricity_sq))
    for _ in range(6):  # dimensionless convergence iteration, not a trajectory timestep
        normal_m = WGS84_A_M / np.sqrt(1. - eccentricity_sq * np.sin(latitude_rad)**2)
        latitude_rad = np.arctan2(z_m + eccentricity_sq * normal_m * np.sin(latitude_rad), horizontal_m)
    altitude_m = (horizontal_m * np.cos(latitude_rad) + z_m * np.sin(latitude_rad)
                  - WGS84_A_M * np.sqrt(1. - eccentricity_sq * np.sin(latitude_rad)**2))
    return latitude_rad, np.arctan2(position_m[..., 1], position_m[..., 0]), altitude_m


def enu_basis(latitude_rad: float, longitude_rad: float) -> FloatArray:
    """Construct a local east/north/up basis in Earth-fixed coordinates.

    Parameters
    ----------
    latitude_rad, longitude_rad : float
        Geodetic latitude and longitude [rad].

    Returns
    -------
    ndarray, shape (3, 3)
        Dimensionless orthonormal matrix with east, north, up columns.
    """
    sin_lat, cos_lat = np.sin(latitude_rad), np.cos(latitude_rad)
    sin_lon, cos_lon = np.sin(longitude_rad), np.cos(longitude_rad)
    return np.array([[-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
                     [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon], [0., cos_lat, sin_lat]])


def rotation_z(angle_rad: float) -> FloatArray:
    """Construct an active right-handed rotation about +z.

    Parameters
    ----------
    angle_rad : float
        Rotation angle [rad].

    Returns
    -------
    ndarray, shape (3, 3)
        Dimensionless rotation matrix.
    """
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]])


def gravity_acceleration(position_m: ArrayLike, *, mu_m3_s2: float = EARTH_MU_M3_S2,
                         j2: float = EARTH_J2) -> FloatArray:
    """Return central plus J2 gravity [m/s^2] in Earth-centered equatorial axes.

    Parameters
    ----------
    position_m : array_like, shape (..., 3)
        Earth-centered position [m], in ECI or Earth-fixed axes.
    mu_m3_s2 : float
        Gravitational parameter [m^3/s^2]; zero supports isolated rocket tests.
    j2 : float
        Dimensionless zonal coefficient; zero selects spherical gravity.

    Returns
    -------
    ndarray, shape (..., 3)
        Gravitational acceleration [m/s^2] in the input frame, without rotation terms.
    """
    position_m = np.asarray(position_m, dtype=float)
    radius_m = np.linalg.norm(position_m, axis=-1, keepdims=True)
    if position_m.shape[-1:] != (3,) or np.any(radius_m <= 0.) or not np.all(np.isfinite(position_m)):
        raise ValueError("Gravity requires finite nonzero 3D radius")
    if not np.isfinite(mu_m3_s2) or mu_m3_s2 < 0. or not np.isfinite(j2):
        raise ValueError("Invalid gravity constants")
    z_ratio_sq = (position_m[..., 2:3] / radius_m)**2
    factors = np.concatenate((5. * z_ratio_sq - 1., 5. * z_ratio_sq - 1., 5. * z_ratio_sq - 3.), axis=-1)
    return (-mu_m3_s2 * position_m / radius_m**3
            + 1.5 * j2 * mu_m3_s2 * WGS84_A_M**2 / radius_m**5 * position_m * factors)


def quaternion_matrix(quaternion: ArrayLike) -> FloatArray:
    """Return body-to-frame rotation from dimensionless scalar-first Hamilton quaternion.

    Parameters
    ----------
    quaternion : array_like, shape (4,)
        Nonzero [w, x, y, z] orientation (dimensionless).

    Returns
    -------
    ndarray, shape (3, 3)
        Active body-to-frame rotation (dimensionless).

    Notes
    -----
    Normalize the local copy for force evaluation, without clipping the ODE state.
    """
    quaternion = np.asarray(quaternion, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)) or np.linalg.norm(quaternion) < 1e-12:
        raise ValueError("Quaternion must be a finite nonzero four-vector")
    w, x, y, z = quaternion / np.linalg.norm(quaternion)  # dimensionless components
    return np.array([[1.-2.*(y*y+z*z), 2.*(x*y-z*w), 2.*(x*z+y*w)],
                     [2.*(x*y+z*w), 1.-2.*(x*x+z*z), 2.*(y*z-x*w)],
                     [2.*(x*z-y*w), 2.*(y*z+x*w), 1.-2.*(x*x+y*y)]])


def quaternion_derivative(quaternion: ArrayLike, relative_body_rate_rad_s: ArrayLike) -> FloatArray:
    """Evaluate Hamilton quaternion kinematics.

    Parameters
    ----------
    quaternion : array_like, shape (4,)
        Body-to-frame [w, x, y, z] quaternion (dimensionless).
    relative_body_rate_rad_s : array_like, shape (3,)
        Body angular velocity relative to the propagation frame, in body axes [rad/s].

    Returns
    -------
    ndarray, shape (4,)
        Quaternion derivative [1/s].
    """
    quaternion = np.asarray(quaternion, dtype=float)
    relative_body_rate_rad_s = np.asarray(relative_body_rate_rad_s, dtype=float)
    return .5 * np.concatenate(([-np.dot(quaternion[1:], relative_body_rate_rad_s)],
                                quaternion[0] * relative_body_rate_rad_s
                                + np.cross(quaternion[1:], relative_body_rate_rad_s)))
