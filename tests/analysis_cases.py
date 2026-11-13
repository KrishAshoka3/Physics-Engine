"""Short deterministic launch fixtures for Phase 3 tests, not realistic vehicle presets."""

import numpy as np

from src.guidance import GuidanceProgram, SteeringConfig
from src.physics import SimulationConfig, StageConfig, VehicleConfig


def short_launch() -> tuple[VehicleConfig, SimulationConfig, GuidanceProgram]:
    """Return a two-stage ~30 kg low-Isp vertical test launch completing within 30 s."""
    vehicle = VehicleConfig("dispersion fixture", (
        StageConfig("booster", 15., 2., 30., 0., mass_flow_kg_s=2., reference_area_m2=.02,
                    stack_length_m=2., thrust_offset_body_m=(-.8, 0., 0.)),
        StageConfig("upper", 8., 1., 35., 0., mass_flow_kg_s=1., reference_area_m2=.02,
                    stack_length_m=1., thrust_offset_body_m=(-.4, 0., 0.))), 3.)
    simulation = SimulationConfig(duration_s=30., max_step_s=.5, rtol=1e-8, rotating_earth=False)
    guidance = GuidanceProgram(SteeringConfig(rail_length_m=.2, vertical_hold_s=.2,
                                             pitch_duration_s=.3, pitch_kick_rad=np.deg2rad(2.)))
    return vehicle, simulation, guidance
