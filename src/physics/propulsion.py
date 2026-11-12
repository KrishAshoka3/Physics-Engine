"""Stage-local vacuum thrust curves and ambient-pressure compensation in SI units."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import G0_M_S2
from .vehicle import StageConfig

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PropulsionState:
    """Broadcast propulsion fields: thrust [N], vacuum thrust [N], positive flow [kg/s]."""

    thrust_n: FloatArray
    vacuum_thrust_n: FloatArray
    mass_flow_kg_s: FloatArray


def propulsion(stage: StageConfig, elapsed_s: ArrayLike, ambient_pressure_pa: ArrayLike,
               *, engine_on: bool = True) -> PropulsionState:
    """Evaluate a stage without using time comparisons to perform engine cutoff.

    Parameters
    ----------
    stage : StageConfig
        Stage properties in SI units.
    elapsed_s : array_like
        Time [s] relative to ignition. Endpoint continuation supports integrator
        trial evaluations; the event scheduler owns the active burn interval.
    ambient_pressure_pa : array_like
        Nonnegative ambient pressure [Pa].
    engine_on : bool
        Discrete mode set by terminal ignition/cutoff events (not a physical unit).

    Returns
    -------
    PropulsionState
        Net force [N], vacuum force [N], and propellant consumption [kg/s].

    Notes
    -----
    F = mdot * Isp_vac * g0 - P * Ae, without silently clipping negative net
    pressure thrust. That regime indicates an invalid/overexpanded nozzle model.
    When off (or a curve has exactly zero flow), nozzle pressure thrust is zero.
    """
    elapsed_s, ambient_pressure_pa = np.broadcast_arrays(np.asarray(elapsed_s, dtype=float), np.asarray(ambient_pressure_pa, dtype=float))
    if not np.all(np.isfinite(elapsed_s)) or not np.all(np.isfinite(ambient_pressure_pa)) or np.any(ambient_pressure_pa < 0.):
        raise ValueError("Require finite time and nonnegative finite pressure")
    if not engine_on:
        zero = np.zeros_like(elapsed_s)
        return PropulsionState(zero, zero.copy(), zero.copy())
    if stage.mass_flow_kg_s is not None:
        flow_kg_s = np.full_like(elapsed_s, stage.mass_flow_kg_s)
        vacuum_n = flow_kg_s * stage.isp_vac_s * G0_M_S2
    else:
        vacuum_n = np.interp(elapsed_s, stage.thrust_curve_time_s, stage.thrust_curve_vac_n)
        flow_kg_s = vacuum_n / (stage.isp_vac_s * G0_M_S2)
    thrust_n = vacuum_n - np.where(flow_kg_s > 0., ambient_pressure_pa * stage.nozzle_exit_area_m2, 0.)
    return PropulsionState(np.asarray(thrust_n), np.asarray(vacuum_n), np.asarray(flow_kg_s))
