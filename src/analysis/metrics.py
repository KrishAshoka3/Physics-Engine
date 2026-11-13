"""Shared SI trajectory metrics, impact projection, and reproducibility metadata.

Notes
-----
Impact is measured only at a ground-impact terminal event. Range coordinates
use a declared spherical azimuthal-equidistant map, not an ellipsoidal geodesic.
Peak altitude is refined with a kinematic Hermite interpolant; structural load
maxima remain sampled quantities and require step-size convergence checks.
"""

from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np
from numpy.typing import NDArray
import scipy
from scipy.interpolate import CubicHermiteSpline

from src.guidance import OrbitTarget
from src.physics import SimulationResult
from src.physics.constants import EARTH_OMEGA_RAD_S, WGS84_A_M
from src.physics.frames import ecef_to_geodetic, rotation_z

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TrajectoryMetrics:
    """Summary in SI units; missing physical outcomes are NaN, not zero.

    Attributes
    ----------
    apogee_altitude_m, peak_altitude_m : float
        Observed/refined apogee [m], or NaN if no maximum was traversed; highest
        altitude [m] within the simulated horizon is always reported separately.
    max_q_pa, max_abs_axial_g : float
        Sampled dynamic pressure [Pa] and absolute axial load [g0, dimensionless].
    downrange_impact_m, crossrange_impact_m, impact_time_s : float
        Terminal-impact map coordinates [m] and elapsed time [s], or NaN.
    insertion_radius_error_m, insertion_radial_error_m_s, insertion_speed_error_m_s : float
        Actual final-stage cutoff errors [m,m/s,m/s], or NaN if not observed.
    insertion_achieved : bool
        True only if all actual cutoff errors meet target tolerances.
    """

    apogee_altitude_m: float
    peak_altitude_m: float
    max_q_pa: float
    max_abs_axial_g: float
    downrange_impact_m: float = np.nan
    crossrange_impact_m: float = np.nan
    impact_time_s: float = np.nan
    insertion_radius_error_m: float = np.nan
    insertion_radial_error_m_s: float = np.nan
    insertion_speed_error_m_s: float = np.nan
    insertion_achieved: bool = False


def impact_coordinates(latitude_rad: float, longitude_rad: float, launch_latitude_rad: float,
                       launch_longitude_rad: float, azimuth_rad: float) -> FloatArray:
    """Return downrange/right-crossrange [m] in a spherical launch-centered map.

    Parameters
    ----------
    latitude_rad, longitude_rad, launch_latitude_rad, launch_longitude_rad : float
        Impact and launch geodetic latitude/longitude [rad].
    azimuth_rad : float
        Fixed nominal range axis, east of north [rad].

    Returns
    -------
    ndarray, shape (2,)
        Arc length times initial-bearing components [m], using WGS84_A radius.
        The antipode is undefined and raises ValueError.
    """
    if not np.all(np.isfinite([latitude_rad, longitude_rad, launch_latitude_rad, launch_longitude_rad, azimuth_rad])):
        raise ValueError("Impact coordinates must be finite radians")
    delta_rad = longitude_rad-launch_longitude_rad
    east = np.cos(latitude_rad)*np.sin(delta_rad)
    north = (np.cos(launch_latitude_rad)*np.sin(latitude_rad)
             - np.sin(launch_latitude_rad)*np.cos(latitude_rad)*np.cos(delta_rad))
    cosine = (np.sin(launch_latitude_rad)*np.sin(latitude_rad)
              + np.cos(launch_latitude_rad)*np.cos(latitude_rad)*np.cos(delta_rad))
    sine = np.hypot(east, north)
    if sine < 1e-14:
        if cosine < 0.:
            raise ValueError("Impact projection is undefined at the launch antipode")
        return np.zeros(2)
    distance_m = WGS84_A_M*np.arctan2(sine, cosine)
    return distance_m/sine*np.array([east*np.sin(azimuth_rad)+north*np.cos(azimuth_rad),
                                     east*np.cos(azimuth_rad)-north*np.sin(azimuth_rad)])


def trajectory_metrics(result: SimulationResult, range_azimuth_rad: float,
                       target: OrbitTarget | None = None, insertion_stage_index: int = 1) -> TrajectoryMetrics:
    """Extract finite-horizon trajectory, impact, and actual-cutoff metrics.

    Parameters
    ----------
    result : SimulationResult
        Propagated SI states and physical event ledger.
    range_azimuth_rad : float
        Common nominal downrange axis [rad], never each dispersed launch axis.
    target : OrbitTarget or None
        Optional spherical-radius insertion target.
    insertion_stage_index : int
        Required cutoff stage, zero-based (dimensionless); default second stage.
    """
    times_s = result.time_s
    height_m = result.telemetry["altitude_m"]
    positions_m = result.telemetry["position_eci_m"]
    velocities_m_s = result.telemetry["velocity_eci_m_s"]
    # Geodetic up is also the height gradient. Evaluating it in ECI is valid
    # because the ellipsoid is axisymmetric; Earth rotation has zero up component.
    latitude_rad, longitude_rad, _ = ecef_to_geodetic(positions_m)
    up = np.column_stack((np.cos(latitude_rad)*np.cos(longitude_rad),
                          np.cos(latitude_rad)*np.sin(longitude_rad), np.sin(latitude_rad)))
    vertical_m_s = np.sum(up*velocities_m_s, axis=1)
    peak_m, apogee_m = float(np.max(height_m)), np.nan
    if len(times_s) >= 2:
        height = CubicHermiteSpline(times_s, height_m, vertical_m_s)
        roots_s = height.derivative().roots(extrapolate=False)
        roots_s = roots_s[np.isfinite(roots_s) & (roots_s > times_s[0]) & (roots_s < times_s[-1])]
        maxima_s = roots_s[height.derivative(2)(roots_s) < 0.]
        if len(maxima_s):
            apogee_m = float(np.max(height(maxima_s)))
            peak_m = max(peak_m, apogee_m)
    impact: dict[str, float] = {}
    if result.termination == "ground_impact":
        time_s = float(times_s[-1])
        omega_rad_s = EARTH_OMEGA_RAD_S if result.config.rotating_earth else 0.
        lat_rad, lon_rad, _ = ecef_to_geodetic(rotation_z(-omega_rad_s*time_s) @ positions_m[-1])
        ranges_m = impact_coordinates(float(lat_rad), float(lon_rad), result.config.latitude_rad,
                                      result.config.longitude_rad, range_azimuth_rad)
        impact = dict(downrange_impact_m=float(ranges_m[0]), crossrange_impact_m=float(ranges_m[1]), impact_time_s=time_s)
    insertion: dict[str, float | bool] = {}
    cutoffs = [event for event in result.events if event.trigger in ("guided_cutoff", "propellant_depletion", "scheduled_cutoff")]
    if target is not None and cutoffs and cutoffs[-1].stage_index == insertion_stage_index:
        # Match the accepted cutoff sample; positions/velocities do not jump there.
        index = int(np.argmin(np.abs(times_s-cutoffs[-1].time_s)))
        radius_m = float(np.linalg.norm(positions_m[index]))
        radial_m_s = float(np.dot(positions_m[index], velocities_m_s[index])/radius_m)
        tangent_m_s = float(np.linalg.norm(velocities_m_s[index]-radial_m_s*positions_m[index]/radius_m))
        radius_error_m = radius_m-WGS84_A_M-target.altitude_m
        speed_error_m_s = tangent_m_s-target.speed_m_s(result.config.mu_m3_s2)
        insertion = dict(insertion_radius_error_m=radius_error_m, insertion_radial_error_m_s=radial_m_s,
                         insertion_speed_error_m_s=speed_error_m_s,
                         insertion_achieved=abs(radius_error_m) <= target.altitude_tolerance_m
                         and max(abs(radial_m_s), abs(speed_error_m_s)) <= target.velocity_tolerance_m_s)
    max_q_pa = max([float(np.max(result.telemetry["q_pa"]))]
                   + [event.telemetry_before["q_pa"] for event in result.events if event.telemetry_before])
    max_g = max([float(np.max(np.abs(result.telemetry["axial_g"])))]
                + [abs(event.telemetry_before["axial_g"]) for event in result.events if event.telemetry_before])
    return TrajectoryMetrics(apogee_m, peak_m, max_q_pa, max_g, **impact, **insertion)


def provenance() -> dict[str, str | bool]:
    """Return dependency versions and repository revision/dirty flag (no physical units)."""
    metadata: dict[str, str | bool] = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    try:
        root = Path(__file__).resolve().parents[2]
        metadata["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        metadata["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        metadata["git_revision"] = "unavailable"
    return metadata


def write_json(path: str | Path, value: Any) -> None:
    """Write dataclasses/arrays as strict UTF-8 JSON; missing nonfinite values become null.

    Parameters
    ----------
    path : str or Path
        Output filename; parent directory must exist.
    value : object
        Result/configuration tree with SI units carried by field names/metadata.
    """
    def clean(item: Any) -> Any:
        """Recursively normalize a serializable tree without changing physical units."""
        if is_dataclass(item):
            return clean(asdict(item))
        if isinstance(item, np.ndarray):
            return clean(item.tolist())
        if isinstance(item, np.generic):
            return clean(item.item())
        if isinstance(item, dict):
            return {str(key): clean(entry) for key, entry in item.items()}
        if isinstance(item, (tuple, list)):
            return [clean(entry) for entry in item]
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")
