"""Analytical/table environment benchmarks; physical quantities use SI suffixes."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.physics.aerodynamics import AeroConfig, WindProfile, aerodynamic_force, coefficients, dynamic_pressure
from src.physics.atmosphere import atmosphere, geometric_to_geopotential, geopotential_to_geometric
from src.physics.constants import AIR_R_J_KG_K, EARTH_J2, EARTH_MU_M3_S2, WGS84_A_M, WGS84_B_M
from src.physics.frames import ecef_to_geodetic, enu_basis, geodetic_to_ecef, gravity_acceleration


def test_us76_table_boundaries() -> None:
    """Compare geopotential layer bases against COESA/NASA values (pressure within 0.02%)."""
    altitude_m = np.array([0., 11000., 20000., 32000., 47000., 51000., 71000., 84852.])
    expected_temperature_k = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.8673]
    expected_pressure_pa = [101325., 22632.06, 5474.889, 868.0187, 110.9063, 66.93887, 3.95642, .3733836]
    result = atmosphere(altitude_m, altitude_kind="geopotential")
    assert_allclose(result.temperature_k, expected_temperature_k, atol=1e-3, rtol=0.)
    assert_allclose(result.pressure_pa, expected_pressure_pa, rtol=2e-4)
    assert_allclose(result.density_kg_m3[0], 1.225, rtol=2e-6)
    assert_allclose(result.speed_of_sound_m_s[0], 340.294, atol=.001)


def test_us76_geometric_samples() -> None:
    """Check independent geometric-height table points (0.1% pressure/density tolerance)."""
    result = atmosphere([10000., 20000., 50000., 86000.])
    assert_allclose(result.temperature_k, [223.2521, 216.65, 270.65, 186.8673], atol=.003)
    assert_allclose(result.pressure_pa, [26499.9, 5529.31, 79.7787, .37338], rtol=1e-3)
    assert_allclose(result.density_kg_m3, [.41351, .0889098, .0010269, 6.958e-6], rtol=1e-3)


def test_coordinate_roundtrip_and_broadcast() -> None:
    """Check atmosphere coordinate inversion and batch/scalar equivalence in SI units."""
    altitude_m = np.array([[0., 11000., 86000.], [90000., 150000., 500000.]])
    potential_m = geometric_to_geopotential(altitude_m)
    assert_allclose(geopotential_to_geometric(potential_m), altitude_m, atol=1e-9)
    assert_allclose(atmosphere(potential_m, altitude_kind="geopotential").pressure_pa, atmosphere(altitude_m).pressure_pa)
    assert atmosphere(altitude_m).pressure_pa.shape == (2, 3)
    assert_allclose(atmosphere(altitude_m).density_kg_m3.ravel(), [float(atmosphere(height_m).density_kg_m3) for height_m in altitude_m.ravel()])


def test_atmosphere_continuity_and_upper_extension() -> None:
    """Require continuous layers, C1 join at 86 km, positive finite fields through 500 km."""
    for height_m in geopotential_to_geometric([11000., 20000., 32000., 47000., 51000., 71000.]):
        left, right = atmosphere(height_m - .001), atmosphere(height_m + .001)
        assert_allclose(left.temperature_k, right.temperature_k, atol=1e-5)
        assert_allclose(left.pressure_pa, right.pressure_pa, rtol=1e-6)
    for field in ("temperature_k", "pressure_pa", "density_kg_m3", "speed_of_sound_m_s"):
        values = np.array([float(getattr(atmosphere(height_m), field)) for height_m in (85999.9, 86000., 86000.1)])
        slopes = np.diff(values) / .1
        assert_allclose(slopes[0], slopes[1], rtol=2e-4, atol=1e-13)
    result = atmosphere(np.linspace(86000., 500000., 5000))
    for values in (result.temperature_k, result.pressure_pa, result.density_kg_m3, result.speed_of_sound_m_s):
        assert np.all(np.isfinite(values)) and np.all(values > 0.)
    assert np.all(np.diff(result.density_kg_m3) < 0.)
    assert np.all(np.diff(result.pressure_pa) < 0.)
    assert_allclose(result.pressure_pa, result.density_kg_m3 * result.gas_constant_j_kg_k * result.temperature_k)


def test_molecular_weight_correction() -> None:
    """Check the US76 Table 8 correction at geometric 80, 83, and 86 km."""
    result = atmosphere([80000., 83000., 86000.])
    assert_allclose(AIR_R_J_KG_K / result.gas_constant_j_kg_k, [1., .999870, .999579], atol=1e-12)
    assert_allclose(result.speed_of_sound_m_s**2, 1.4 * result.gas_constant_j_kg_k * result.temperature_k)


@pytest.mark.parametrize("altitude_m", [np.nan, np.inf, -5001.])
def test_invalid_atmosphere(altitude_m: float) -> None:
    """Invalid altitude [m] must fail explicitly."""
    with pytest.raises(ValueError):
        atmosphere(altitude_m)


def test_wind_dynamic_pressure_and_lift_geometry() -> None:
    """Resolve sheared 3D winds, oppose relative velocity with drag, and keep lift orthogonal."""
    wind = WindProfile((0., 1000.), ((0., 0., 0.), (20., -10., 4.)))
    wind_m_s = wind.velocity(500.)
    assert_allclose(wind_m_s, [10., -5., 2.])
    assert_allclose(dynamic_pressure(1.2, [110., -5., 2.], wind_m_s), 6000.)
    force_n, mach, q_pa, alpha_rad = aerodynamic_force([100., 0., 0.], [np.cos(.1), np.sin(.1), 0.], 1.2, 340., 2.)
    cd, cl = coefficients(100./340., .1)
    assert_allclose(force_n, [-6000.*2.*cd, 6000.*2.*cl, 0.], atol=1e-9)
    assert_allclose([mach, q_pa, alpha_rad], [100./340., 6000., .1])
    zero_n, _, zero_q_pa, _ = aerodynamic_force([0., 0., 0.], [1., 0., 0.], 1.2, 340., 2.)
    assert_allclose(zero_n, 0.)
    assert zero_q_pa == 0.


def test_continuous_aero_parity_wave_drag_and_newtonian_limit() -> None:
    """Require transonic divergence, no branch jumps, correct alpha parity and high-Mach limit."""
    mach = np.linspace(0., 10., 10001)
    cd, cl = coefficients(mach, .12)
    assert np.max(np.abs(np.diff(cd))) < .004
    assert np.all(cd > 0.)
    assert float(coefficients(1.15, 0.)[0]) > float(coefficients(.7, 0.)[0]) * 2.
    assert_allclose(cd, coefficients(mach, -.12)[0])
    assert_allclose(cl, -coefficients(mach, -.12)[1])
    cd_inf, cl_inf = coefficients(10000., .3)
    assert_allclose(cd_inf, .15 + 2.*np.sin(.3)**2, rtol=1e-6)
    assert_allclose(cl_inf, 2.*np.sin(.3)**2*np.cos(.3), rtol=1e-6)
    assert float(coefficients(.5, .1)[1]) > float(coefficients(10., .1)[1])
    with pytest.raises(ValueError):
        coefficients(-1., 0.)
    with pytest.raises(ValueError):
        AeroConfig(base_cd=-.1)
    with pytest.raises(ValueError):
        WindProfile((0., 0.), ((0., 0., 0.), (0., 0., 0.)))


@pytest.mark.parametrize("latitude_rad", [0., .5, -1.2, np.pi/2., -np.pi/2.])
def test_wgs84_roundtrip_and_enu(latitude_rad: float) -> None:
    """Recover geodetic position to sub-mm accuracy, including poles and 500 km height."""
    for altitude_m in (-500., 0., 500000.):
        position_m = geodetic_to_ecef(latitude_rad, .7, altitude_m)
        latitude_back_rad, longitude_back_rad, altitude_back_m = ecef_to_geodetic(position_m)
        assert_allclose(latitude_back_rad, latitude_rad, atol=1e-12)
        assert_allclose(longitude_back_rad, .7, atol=1e-12)
        assert_allclose(altitude_back_m, altitude_m, atol=1e-5)
    basis = enu_basis(latitude_rad, .7)
    assert_allclose(basis.T @ basis, np.eye(3), atol=1e-15)
    assert_allclose(np.linalg.det(basis), 1.)


def test_central_j2_gravity_at_equator_and_pole() -> None:
    """Verify known J2 equator/pole signs and absolute accelerations [m/s^2]."""
    equatorial_m_s2 = EARTH_MU_M3_S2 / WGS84_A_M**2 * (1. + 1.5 * EARTH_J2)
    polar_m_s2 = EARTH_MU_M3_S2 / WGS84_B_M**2 * (1. - 3.*EARTH_J2*(WGS84_A_M/WGS84_B_M)**2)
    assert_allclose(gravity_acceleration([WGS84_A_M, 0., 0.]), [-equatorial_m_s2, 0., 0.], atol=1e-12)
    assert_allclose(gravity_acceleration([0., 0., WGS84_B_M]), [0., 0., -polar_m_s2], atol=1e-12)
    assert_allclose(gravity_acceleration([WGS84_A_M, 0., 0.], j2=0.), [-EARTH_MU_M3_S2/WGS84_A_M**2, 0., 0.])


def test_j2_force_matches_potential_gradient() -> None:
    """Off-axis gravity must match an independent numerical potential gradient to 2e-8 m/s^2."""
    position_m = np.array([6e6, 2e6, 3e6])
    gradient_m_s2 = np.empty(3)
    for axis in range(3):
        delta_m = np.eye(3)[axis] * .5
        gradient_m_s2[axis] = (_potential_m2_s2(position_m + delta_m) - _potential_m2_s2(position_m - delta_m)) / 1.
    assert_allclose(gravity_acceleration(position_m), -gradient_m_s2, atol=2e-8, rtol=0.)


def _potential_m2_s2(position_m: np.ndarray) -> float:
    """Evaluate independent central+J2 specific potential [m^2/s^2]."""
    radius_m = np.linalg.norm(position_m)
    legendre_p2 = .5*(3.*(position_m[2]/radius_m)**2 - 1.)
    return float(-EARTH_MU_M3_S2/radius_m * (1.-EARTH_J2*(WGS84_A_M/radius_m)**2*legendre_p2))
