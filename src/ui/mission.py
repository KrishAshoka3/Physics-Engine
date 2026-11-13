"""Shared mission setup and presentation data, without duplicate force models.

Notes
-----
All model inputs remain SI. Plot labels convert to km/kPa/degrees explicitly.
Ground tracks use the analysis API's azimuthal-equidistant projection; ECI
epoch is launch and geographic coordinates rotate at the configured Earth rate.
"""

from dataclasses import asdict, dataclass, replace
import json

import numpy as np
import pandas as pd

from src.analysis import (DesignPoint, DispersionSpace, OptimizationConfig,
                          default_dispersion_space, evaluate_candidate, trajectory_metrics)
from src.analysis.metrics import impact_coordinates, provenance
from src.guidance import GuidanceProgram, OrbitTarget, SteeringConfig
from src.physics import PRESETS, SimulationConfig, SimulationResult, VehicleConfig
from src.physics.constants import EARTH_OMEGA_RAD_S
from src.physics.frames import ecef_to_geodetic


@dataclass(frozen=True)
class MissionSettings:
    """User controls with SI units, dimensionless scales and model selectors.

    Parameters
    ----------
    preset : str
        Exact public vehicle preset key.
    thrust_scale, fuel_scale, dry_scale : float
        Positive dimensionless factors, common to both stages.
    coast_s, pitch_rad, target_altitude_m : float
        Interstage coast [s], pitch kick [rad], circular target altitude [m].
    rotating_earth, tvc : bool
        Earth rotation and closed-loop 6DoF switches (dimensionless).
    """

    preset: str = next(iter(PRESETS))
    thrust_scale: float = 1.
    fuel_scale: float = 1.
    dry_scale: float = 1.
    coast_s: float = 6.
    pitch_rad: float = float(np.deg2rad(5.))
    target_altitude_m: float = 200000.
    rotating_earth: bool = True
    tvc: bool = False


def build_mission(settings: MissionSettings) -> tuple[VehicleConfig, SimulationConfig, GuidanceProgram]:
    """Build immutable public configurations from validated SI controls.

    Returns
    -------
    vehicle, simulation, guidance : tuple
        Two-stage launch; sounding missions run to impact or 1000 s. Orbital
        missions use closed-loop insertion and a 600 s observation horizon.
    """
    values = (settings.thrust_scale, settings.fuel_scale, settings.dry_scale, settings.coast_s,
              settings.pitch_rad, settings.target_altitude_m)
    if not np.all(np.isfinite(values)) or min(values[:3]) <= 0. or min(values[3:]) < 0.:
        raise ValueError("Mission scales must be positive and SI controls finite/nonnegative")
    vehicle = PRESETS[settings.preset]
    stages = tuple(replace(stage, mass_flow_kg_s=stage.mass_flow_kg_s*settings.thrust_scale,
                            propellant_mass_kg=stage.propellant_mass_kg*settings.fuel_scale,
                            dry_mass_kg=stage.dry_mass_kg*settings.dry_scale,
                            coast_after_separation_s=settings.coast_s if index == 0 else 0.)
                   for index, stage in enumerate(vehicle.stages))
    orbital = "Orbital" in settings.preset
    config = SimulationConfig(duration_s=600. if orbital else 1000., max_step_s=2.,
                               rotating_earth=settings.rotating_earth, dof=6 if settings.tvc else 3)
    guidance = GuidanceProgram(SteeringConfig(pitch_kick_rad=settings.pitch_rad,
        target=OrbitTarget(settings.target_altitude_m) if orbital else None))
    return replace(vehicle, stages=stages), config, guidance


def telemetry_frame(result: SimulationResult, azimuth_rad: float = np.pi/2.) -> pd.DataFrame:
    """Flatten every telemetry/state component and derive geographic plotting columns.

    Parameters
    ----------
    result : SimulationResult
        Right-continuous sampled state and event ledger.
    azimuth_rad : float
        Nominal bearing east of north [rad]; positive crossrange is to its right.

    Returns
    -------
    pandas.DataFrame
        All SI telemetry, state columns, event labels, geodetic coordinates,
        projected range and Earth-relative geodetic flight path angle [rad].
        Stationary flight path angle is undefined (NaN).
    """
    columns = {"time_s": result.time_s, "stage_index": result.stage_index, "engine_on": result.engine_on}
    for name, array in result.telemetry.items():
        if array.ndim == 1:
            columns[name] = array
        else:
            for index in range(array.shape[1]):
                columns[f"{name}_{index}"] = array[:, index]
    for index in range(result.state.shape[1]):
        columns[f"state_{index}"] = result.state[:, index]
    omega_rad_s = EARTH_OMEGA_RAD_S if result.config.rotating_earth else 0.
    angle_rad = result.time_s*omega_rad_s
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    positions_m = result.telemetry["position_eci_m"].copy()
    velocities_m_s = result.telemetry["velocity_eci_m_s"]-np.cross([0., 0., omega_rad_s], positions_m)
    for vectors in (positions_m, velocities_m_s):
        east_x, north_y = vectors[:, 0].copy(), vectors[:, 1].copy()
        vectors[:, 0], vectors[:, 1] = cosine*east_x+sine*north_y, -sine*east_x+cosine*north_y
    latitude_rad, longitude_rad, _ = ecef_to_geodetic(positions_m)
    up = np.column_stack((np.cos(latitude_rad)*np.cos(longitude_rad),
                          np.cos(latitude_rad)*np.sin(longitude_rad), np.sin(latitude_rad)))
    speed_m_s = np.linalg.norm(velocities_m_s, axis=1)
    vertical_m_s = np.sum(up*velocities_m_s, axis=1)
    ratio = np.divide(vertical_m_s, speed_m_s, out=np.full_like(speed_m_s, np.nan), where=speed_m_s > 1e-8)
    columns["flight_path_angle_rad"] = np.arcsin(np.clip(ratio, -1., 1.))
    columns["earth_relative_speed_m_s"] = speed_m_s
    columns["latitude_rad"], columns["longitude_rad"] = latitude_rad, longitude_rad
    ranges_m = np.array([impact_coordinates(float(lat), float(lon), result.config.latitude_rad,
        result.config.longitude_rad, azimuth_rad) for lat, lon in zip(latitude_rad, longitude_rad)])
    columns["downrange_m"], columns["crossrange_m"] = ranges_m.T
    frame = pd.DataFrame(columns)
    frame["events"] = ""
    for event in result.events:
        index = int(np.argmin(abs(result.time_s-event.time_s)))
        frame.loc[index, "events"] += (";" if frame.loc[index, "events"] else "")+event.name
    return frame


def mission_export(settings: MissionSettings) -> str:
    """Return replayable JSON with SI configs, state schema, versions and Git revision."""
    vehicle, config, guidance = build_mission(settings)
    return json.dumps(dict(settings=asdict(settings), vehicle=asdict(vehicle), simulation=asdict(config),
        guidance=asdict(guidance), provenance=provenance(),
        state_schema="position[0:3] m; velocity[3:6] m/s in simulation.frame; mass[6], fuel[7] kg; "
                     "6DoF: q_wxyz[8:12] body-to-frame; omega_body[12:15] rad/s; gimbal[15:17] rad",
        telemetry_schema="SI suffixes; Mach, twr, axial_g in g0 and quaternion are dimensionless; "
                         "ECI epoch=launch; time series right-continuous; NaN denotes undefined"), indent=2)


def scaled_uncertainty(scale: float) -> DispersionSpace:
    """Scale documented dispersion half-ranges by a positive dimensionless multiplier."""
    if not np.isfinite(scale) or scale <= 0.:
        raise ValueError("Uncertainty multiplier must be positive")
    space = default_dispersion_space()
    return replace(space, parameters=tuple(replace(p, half_range=p.half_range*scale) for p in space.parameters))


def trade_study(settings: MissionSettings, limits: OptimizationConfig, variable: str,
                points: int = 7) -> pd.DataFrame:
    """Run a one-variable SI trade while holding other design inputs fixed.

    Notes
    -----
    Curves are actual simulations, not interpolated optimizer claims. Structural
    constraints cover the complete simulated mission, including ballistic descent.
    """
    if variable not in ("fuel split", "pitch") or points < 2:
        raise ValueError("Select fuel split or pitch and at least two points")
    vehicle, config, guidance = build_mission(settings)
    fraction = vehicle.stages[0].propellant_mass_kg/sum(s.propellant_mass_kg for s in vehicle.stages)
    values = np.linspace(*(limits.stage1_fraction_bounds if variable == "fuel split" else limits.pitch_kick_bounds_rad), points)
    rows = []
    for value in values:
        design = DesignPoint(float(value) if variable == "fuel split" else fraction,
                             float(value) if variable == "pitch" else settings.pitch_rad, vehicle.payload_mass_kg)
        evaluated = evaluate_candidate(vehicle, config, guidance, limits, design)
        rows.append(dict(value=float(value), objective=evaluated.objective_value, feasible=evaluated.feasible,
                          max_q_pa=evaluated.metrics.max_q_pa if evaluated.metrics else np.nan,
                          max_axial_g=evaluated.metrics.max_abs_axial_g if evaluated.metrics else np.nan,
                          error=evaluated.error))
    return pd.DataFrame(rows)
