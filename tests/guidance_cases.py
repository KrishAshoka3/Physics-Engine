"""Shared SI fixtures for guidance/controller verification, not flight calibration."""

from dataclasses import replace
from typing import Literal

import numpy as np

from src.guidance import AutopilotConfig, GuidanceProgram, SteeringConfig
from src.physics import SimulationConfig, WindProfile
from src.physics.constants import EARTH_MU_M3_S2, G0_M_S2, WGS84_A_M
from src.physics.flight import NavigationState
from src.physics.frames import enu_basis
from src.physics.vehicle import StageConfig, VehicleConfig


def upper_vehicle() -> VehicleConfig:
    """Return a 1000 kg upper-stage test vehicle with 300 kg fuel and a 2 m TVC arm."""
    return VehicleConfig("upper test", (StageConfig("upper", 700., 300., 340., 0., mass_flow_kg_s=3.,
                         thrust_offset_body_m=(-2., 0., 0.), stack_length_m=6.),), 0.)


def navigation_sample() -> NavigationState:
    """Return an equatorial ECI upper-stage sample at 196 km, vr=200, vt=7400 m/s."""
    stage = upper_vehicle().stages[0]
    return NavigationState(0., np.array([WGS84_A_M+196000., 0., 0.]), np.array([200., 7400., 0.]),
                           np.array([200., 7400., 0.]), enu_basis(0., 0.), enu_basis(0., 0.),
                           np.eye(3), np.zeros(3), np.zeros(2), 1000., 300., 196000.,
                           3.*340.*G0_M_S2, 3., 1000.*np.asarray(stage.principal_inertia_per_mass_m2),
                           0, 0., True, False, EARTH_MU_M3_S2)


def wind_tracking_case(mode: Literal["PID", "LQR"], altitude_m: float = 15000.) -> tuple[VehicleConfig, SimulationConfig, GuidanceProgram]:
    """Return a 25 s three-axis wind-shear TVC case starting at altitude_m [m]."""
    vehicle = upper_vehicle()
    vehicle = replace(vehicle, stages=(replace(vehicle.stages[0], reference_area_m2=.05, cp_offset_body_m=(-.5, 0., 0.)),))
    wind = WindProfile(tuple(altitude_m+np.array([0., 500., 1000., 1500., 2000., 3000.])),
                       ((0., 0., 0.), (60., 20., 4.), (-40., -30., -4.), (40., 0., 2.), (0., 0., 0.), (0., 0., 0.)))
    config = SimulationConfig(duration_s=25., dof=6, launch_altitude_m=altitude_m, initial_elevation_rad=np.deg2rad(85.),
                              initial_velocity_enu_m_s=(0., 0., 300.), rotating_earth=False, wind=wind)
    program = GuidanceProgram(SteeringConfig(rail_length_m=0., vertical_hold_s=0., pitch_duration_s=6., pitch_kick_rad=np.deg2rad(3.)),
                              AutopilotConfig(mode=mode))
    return vehicle, config, program
