"""Consolidated Phase 1 acceptance tests.

Notes
-----
Reference: US76 (NASA-TM-X-74335), Tables 4/8 and main tables,
https://ntrs.nasa.gov/citations/19770009539 . ICAO-compatible layer heights
are geopotential; 86 km is geometric and uses the US76 kinetic-temperature
correction. Rounded published values require explicit rounding tolerances,
not floating-point equality. All physical inputs below are SI.
"""
from dataclasses import replace
from typing import Literal

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.physics import SimulationConfig, StageConfig, VehicleConfig, atmosphere, propulsion, simulate
from src.physics.atmosphere import geopotential_to_geometric
from src.physics.constants import G0_M_S2


@pytest.mark.parametrize('height_m,kind,temperature_k,pressure_pa,density_kg_m3,sound_m_s', [
    (0., 'geopotential', 288.15, 101325., 1.225, 340.294),
    (11000., 'geopotential', 216.65, 22632.06, .363918, 295.069),
    (20000., 'geopotential', 216.65, 5474.889, .0880349, 295.069),
    (47000., 'geopotential', 270.65, 110.9063, .00142753, 329.799),
    (86000., 'geometric', 186.8673, .37338, 6.95782e-6, 274.096),
])
def test_reference_atmosphere(height_m: float, kind: Literal['geometric', 'geopotential'],
                              temperature_k: float, pressure_pa: float,
                              density_kg_m3: float, sound_m_s: float) -> None:
    """Match independently tabulated T [K], P [Pa], rho [kg/m^3], and a [m/s]."""
    result = atmosphere(height_m, altitude_kind=kind)
    assert_allclose(result.temperature_k, temperature_k, atol=.0001, rtol=0.)
    assert_allclose(result.pressure_pa, pressure_pa, rtol=2e-5)
    assert_allclose(result.density_kg_m3, density_kg_m3, rtol=5e-6)
    assert_allclose(result.speed_of_sound_m_s, sound_m_s, atol=.001, rtol=0.)
    geometric_m = float(geopotential_to_geometric(height_m)) if kind == 'geopotential' else height_m
    assert_allclose(atmosphere(geometric_m).pressure_pa, result.pressure_pa, rtol=1e-12)


def _vehicle() -> VehicleConfig:
    """Return a 120 kg single-stage analytical test vehicle with 100 kg propellant."""
    return VehicleConfig('acceptance', (StageConfig('motor', 10., 100., 300., .02, mass_flow_kg_s=10.),), 10.)


def _vacuum(method: Literal['DOP853', 'RK45'], rtol: float = 1e-9) -> SimulationConfig:
    """Return vacuum, zero-gravity, nonrotating inputs; rtol is dimensionless."""
    return SimulationConfig(duration_s=12., launch_altitude_m=100., atmosphere_enabled=False,
                            rotating_earth=False, mu_m3_s2=0., j2_enabled=False,
                            method=method, rtol=rtol, max_step_s=10., velocity_atol_m_s=1e-10)


@pytest.mark.parametrize('pressure_pa', [101325., 22632.06, 5474.889])
def test_nozzle_compensation(pressure_pa: float) -> None:
    """Verify vacuum thrust exceeds atmospheric thrust by exactly P*Ae [N]."""
    stage = _vehicle().stages[0]
    vacuum_n = float(propulsion(stage, 0., 0.).thrust_n)
    ambient_n = float(propulsion(stage, 0., pressure_pa).thrust_n)
    assert vacuum_n > ambient_n > 0.
    assert_allclose(vacuum_n-ambient_n, pressure_pa*.02, rtol=1e-12)
    assert_allclose(vacuum_n, 10.*300.*G0_M_S2)
    assert float(propulsion(stage, 0., pressure_pa, engine_on=False).thrust_n) == 0.


@pytest.mark.parametrize('dof', [3, 6])
@pytest.mark.parametrize('residual', [False, True])
def test_multistage_mass_budget(dof: Literal[3, 6], residual: bool) -> None:
    """Close initial mass = exhaust + separated mass + fairing + final mass [kg]."""
    booster = replace(_vehicle().stages[0], cutoff_time_s=5. if residual else None,
                      separation_delay_s=1., coast_after_separation_s=2.)
    upper = StageConfig('upper', 5., 20., 320., 0., mass_flow_kg_s=10.)
    vehicle = VehicleConfig('budget', (booster, upper), 2., 3., fairing_jettison_time_s=3.)
    result = simulate(vehicle, replace(_vacuum('DOP853'), duration_s=20., dof=dof))
    separation = next(event for event in result.events if event.name == 'stage_1_separation')
    separated_kg = 10. + (50. if residual else 0.)
    exhaust_kg = (50. if residual else 100.) + 20.
    assert_allclose(separation.state_before[6]-separation.state_after[6], separated_kg, atol=1e-7)
    assert_allclose(result.state[-1, 6], 7., atol=1e-7)
    assert_allclose(vehicle.initial_mass_kg, exhaust_kg+separated_kg+3.+result.state[-1, 6], atol=1e-7)
    assert np.all(np.diff(result.state[:, 6]) <= 1e-7)
    assert np.all(result.state[:, 7] >= -1e-7)
    for event in result.events:
        assert_allclose(event.state_before[:6], event.state_after[:6], atol=0.)
    assert len([event for event in result.events if event.name == 'fairing_jettison']) == 1


@pytest.mark.parametrize('method', ['DOP853', 'RK45'])
@pytest.mark.parametrize('dof', [3, 6])
def test_tsiolkovsky_acceptance(method: Literal['DOP853', 'RK45'], dof: Literal[3, 6]) -> None:
    """Require ideal vacuum delta-V relative error strictly below 0.01 percent."""
    result = simulate(_vehicle(), replace(_vacuum(method), dof=dof))
    exact_m_s = 300.*G0_M_S2*np.log(120./20.)
    actual_m_s = np.linalg.norm(result.state[-1, 3:6]-result.state[0, 3:6])
    assert abs(actual_m_s/exact_m_s-1.) < 1e-4
    assert_allclose(result.events[0].time_s, 10., atol=1e-8)
    assert np.all(result.telemetry['q_pa'] == 0.)


@pytest.mark.parametrize('method', ['DOP853', 'RK45'])
def test_requested_tolerance_convergence(method: Literal['DOP853', 'RK45']) -> None:
    """Compare dimensionless rtol 1e-6 and 1e-9 against an independent exact solution."""
    loose = simulate(_vehicle(), _vacuum(method, 1e-6))
    tight = simulate(_vehicle(), _vacuum(method, 1e-9))
    exact_m_s = 300.*G0_M_S2*np.log(6.)
    loose_error_m_s = abs(loose.state[-1, 3]-exact_m_s)
    tight_error_m_s = abs(tight.state[-1, 3]-exact_m_s)
    assert tight_error_m_s < loose_error_m_s
    assert tight_error_m_s/exact_m_s < 1e-4
    assert_allclose(loose.events[0].time_s, tight.events[0].time_s, atol=1e-8)
