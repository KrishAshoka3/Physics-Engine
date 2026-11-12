"""Vectorized seven-layer COESA atmosphere and a C1 exponential surrogate.

Notes
-----
The seven-layer model includes the US76 Table 8 molecular-weight correction
from 80 to 86 km and returns kinetic temperature. Above 86 km this is a numerical
surrogate, NOT the full US76 diffusive-equilibrium or space-weather atmosphere.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import PchipInterpolator

from .constants import AIR_GAMMA, AIR_R_J_KG_K, ATMOSPHERE_RADIUS_M, G0_M_S2

FloatArray = NDArray[np.float64]
LAYER_HEIGHTS_M: FloatArray = np.array([0., 11000., 20000., 32000., 47000., 51000., 71000.])
LAPSE_RATES_K_M: FloatArray = np.array([-.0065, 0., .001, .0028, 0., -.0028, -.002])
# US76 Table 8 geometric-height molecular-weight ratios (dimensionless).
_MOLECULAR_RATIO = PchipInterpolator(
    np.arange(80000., 86001., 500.),
    [1., .999996, .999989, .999971, .999941, .999909, .999870,
     .999829, .999786, .999741, .999694, .999641, .999579])
_RATIO_SLOPE_PER_M: float = float(_MOLECULAR_RATIO.derivative()(86000.))


def geometric_to_geopotential(altitude_m: ArrayLike) -> FloatArray:
    """Convert geometric height to COESA geopotential height.

    Parameters
    ----------
    altitude_m : array_like
        Geometric height [m], above the radius singularity.

    Returns
    -------
    ndarray
        Geopotential height [m], with the input shape.
    """
    altitude_m = np.asarray(altitude_m, dtype=float)
    if np.any(~np.isfinite(altitude_m)) or np.any(altitude_m <= -ATMOSPHERE_RADIUS_M):
        raise ValueError("Geometric altitude must be finite and above the radius singularity.")
    return np.asarray(ATMOSPHERE_RADIUS_M * altitude_m / (ATMOSPHERE_RADIUS_M + altitude_m))


def geopotential_to_geometric(altitude_m: ArrayLike) -> FloatArray:
    """Convert geopotential height to geometric height.

    Parameters
    ----------
    altitude_m : array_like
        Geopotential height [m], below the COESA reference radius.

    Returns
    -------
    ndarray
        Geometric height [m], with the input shape.
    """
    altitude_m = np.asarray(altitude_m, dtype=float)
    if np.any(~np.isfinite(altitude_m)) or np.any(altitude_m >= ATMOSPHERE_RADIUS_M):
        raise ValueError("Geopotential altitude must be finite and below the reference radius.")
    return np.asarray(ATMOSPHERE_RADIUS_M * altitude_m / (ATMOSPHERE_RADIUS_M - altitude_m))


def _layer_bases() -> tuple[FloatArray, FloatArray]:
    """Recursively derive layer-base temperature [K] and pressure [Pa]."""
    temperature_k = np.empty(7)
    pressure_pa = np.empty(7)
    temperature_k[0], pressure_pa[0] = 288.15, 101325.0
    for index in range(1, 7):  # dimensionless layer index; only six setup operations
        delta_m = LAYER_HEIGHTS_M[index] - LAYER_HEIGHTS_M[index - 1]
        lapse_k_m = LAPSE_RATES_K_M[index - 1]
        temperature_k[index] = temperature_k[index - 1] + lapse_k_m * delta_m
        if lapse_k_m == 0.:
            pressure_pa[index] = pressure_pa[index - 1] * np.exp(
                -G0_M_S2 * delta_m / (AIR_R_J_KG_K * temperature_k[index - 1]))
        else:
            pressure_pa[index] = pressure_pa[index - 1] * (
                temperature_k[index] / temperature_k[index - 1]
            ) ** (-G0_M_S2 / (AIR_R_J_KG_K * lapse_k_m))
    return temperature_k, pressure_pa


BASE_TEMPERATURE_K, BASE_PRESSURE_PA = _layer_bases()


@dataclass(frozen=True)
class AtmosphereState:
    """Atmospheric fields with the input broadcast shape (0-D for scalar input).

    Attributes
    ----------
    temperature_k : ndarray
        Kinetic temperature [K] below 86 km; surrogate temperature above.
    pressure_pa : ndarray
        Static pressure [Pa].
    density_kg_m3 : ndarray
        Mass density [kg/m^3].
    speed_of_sound_m_s : ndarray
        Perfect-gas sound speed [m/s]; diagnostic only in rarefied flow.
    gas_constant_j_kg_k : ndarray
        Local specific gas constant [J/(kg*K)], including the Table 8 correction.
    """

    temperature_k: FloatArray
    pressure_pa: FloatArray
    density_kg_m3: FloatArray
    speed_of_sound_m_s: FloatArray
    gas_constant_j_kg_k: FloatArray


def atmosphere(altitude_m: ArrayLike, *, altitude_kind: Literal["geometric", "geopotential"] = "geometric") -> AtmosphereState:
    """Evaluate atmospheric properties without Python loops over altitude.

    Parameters
    ----------
    altitude_m : array_like
        Altitude [m]. Nominal domain is 0--500000 m geometric. The troposphere
        is extended to -5000 m for below-datum sites and root-finder trial states.
    altitude_kind : {"geometric", "geopotential"}
        Input coordinate convention (dimensionless selector).

    Returns
    -------
    AtmosphereState
        Temperature [K], pressure [Pa], density [kg/m^3], sound speed [m/s].

    Raises
    ------
    ValueError
        If altitude is nonfinite, below -5000 m geometric, or the kind is invalid.

    Notes
    -----
    Above 86 km, log-density relaxes exponentially from the boundary gradient
    to a 50000 m scale height; temperature approaches 1000 K. Values and first
    derivatives match at 86 km. Continuation beyond 500 km is mathematical only.
    Positive sound speed is retained even when density underflows to vacuum.
    """
    if altitude_kind not in ("geometric", "geopotential"):
        raise ValueError("altitude_kind must be geometric or geopotential")
    geometric_m = (np.asarray(altitude_m, dtype=float) if altitude_kind == "geometric"
                   else geopotential_to_geometric(altitude_m))
    if np.any(~np.isfinite(geometric_m)) or np.any(geometric_m < -5000.):
        raise ValueError("Atmosphere requires finite geometric altitude >= -5000 m")
    lower_m = np.minimum(geometric_m, 86000.)
    potential_m = geometric_to_geopotential(lower_m)
    index = np.clip(np.searchsorted(LAYER_HEIGHTS_M, potential_m, side="right") - 1, 0, 6)
    delta_m = potential_m - LAYER_HEIGHTS_M[index]
    lapse_k_m = LAPSE_RATES_K_M[index]
    base_k = BASE_TEMPERATURE_K[index]
    temperature_k = base_k + lapse_k_m * delta_m
    # Safe denominators avoid evaluating a divide-by-zero in the unused branch.
    nonzero_lapse_k_m = np.where(lapse_k_m == 0., 1., lapse_k_m)
    log_pressure_ratio = np.where(
        lapse_k_m == 0., -G0_M_S2 * delta_m / (AIR_R_J_KG_K * base_k),
        -G0_M_S2 / (AIR_R_J_KG_K * nonzero_lapse_k_m) * np.log(temperature_k / base_k))
    pressure_pa = BASE_PRESSURE_PA[index] * np.exp(log_pressure_ratio)
    density_kg_m3 = pressure_pa / (AIR_R_J_KG_K * temperature_k)
    molecular_ratio = _MOLECULAR_RATIO(np.clip(geometric_m, 80000., 86000.))
    if np.any(geometric_m > 86000.):
        delta_upper_m = np.maximum(geometric_m - 86000., 0.)
        boundary_factor = (ATMOSPHERE_RADIUS_M / (ATMOSPHERE_RADIUS_M + 86000.)) ** 2
        slope_k_m = LAPSE_RATES_K_M[-1] * boundary_factor
        inverse_scale_m = (G0_M_S2 * boundary_factor / AIR_R_J_KG_K + slope_k_m) / temperature_k
        relaxation = np.exp(-delta_upper_m / 60000.)
        upper_density_kg_m3 = density_kg_m3 * np.exp(
            -delta_upper_m / 50000. - (inverse_scale_m - 1. / 50000.) * 60000. * (1. - relaxation))
        # Extend the tiny composition correction with a matching first derivative.
        # This regularization is not a thermospheric species-composition model.
        upper_ratio = .999579 + _RATIO_SLOPE_PER_M * 10000. * (1. - np.exp(-delta_upper_m / 10000.))
        limiting_ratio = .999579 + _RATIO_SLOPE_PER_M * 10000.
        scaled_height = delta_upper_m / 60000.
        upper_temperature_k = temperature_k + (1000. / limiting_ratio - temperature_k) * (
            1. - (1. + scaled_height) * np.exp(-scaled_height)
        ) + slope_k_m * delta_upper_m * np.exp(-scaled_height)
        upper_mask = geometric_m > 86000.
        molecular_ratio = np.where(upper_mask, upper_ratio, molecular_ratio)
        temperature_k = np.where(upper_mask, upper_temperature_k, temperature_k)
        density_kg_m3 = np.where(upper_mask, upper_density_kg_m3, density_kg_m3)
        pressure_pa = density_kg_m3 * AIR_R_J_KG_K * temperature_k
    return AtmosphereState(np.asarray(temperature_k * molecular_ratio), np.asarray(pressure_pa),
                           np.asarray(density_kg_m3), np.sqrt(AIR_GAMMA * AIR_R_J_KG_K * temperature_k),
                           np.asarray(AIR_R_J_KG_K / molecular_ratio))
