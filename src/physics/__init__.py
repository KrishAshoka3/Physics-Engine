"""SI core physics API; see ``docs/physics_models.md`` for validity and conventions."""

from .aerodynamics import AeroConfig, WindProfile, coefficients, dynamic_pressure
from .atmosphere import AtmosphereState, atmosphere
from .propulsion import PropulsionState, propulsion
from .simulator import ControlInput, SimulationConfig, SimulationResult, initial_state, orbital_elements, simulate
from .vehicle import PRESETS, StageConfig, VehicleConfig

__all__ = ["AeroConfig", "WindProfile", "coefficients", "dynamic_pressure", "AtmosphereState",
           "atmosphere", "PropulsionState", "propulsion", "ControlInput", "SimulationConfig",
           "SimulationResult", "initial_state", "orbital_elements", "simulate", "PRESETS",
           "StageConfig", "VehicleConfig"]
