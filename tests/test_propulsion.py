"""Thrust, mass bookkeeping, and preset input validation in SI units."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.physics.constants import G0_M_S2
from src.physics.propulsion import propulsion
from src.physics.vehicle import PRESETS, StageConfig, VehicleConfig


def test_pressure_compensation_and_engine_off() -> None:
    """Verify F=mdot*Isp*g0-P*Ae [N], with no pressure thrust after shutdown."""
    stage = StageConfig("test", 10., 100., 300., .02, mass_flow_kg_s=5.)
    result = propulsion(stage, [0., 5.], [0., 101325.])
    assert_allclose(result.thrust_n, 5.*300.*G0_M_S2 - np.array([0., 101325.])*.02)
    assert_allclose(result.mass_flow_kg_s, 5.)
    assert_allclose(propulsion(stage, 21., 101325., engine_on=False).thrust_n, 0.)
    assert_allclose(propulsion(stage, 21., 101325., engine_on=False).mass_flow_kg_s, 0.)


def test_curve_interpolation_and_integrated_impulse() -> None:
    """Triangle thrust curve integrates to known impulse [N*s] and fuel mass [kg]."""
    stage = StageConfig("curve", 10., 10., 200., 0., thrust_curve_time_s=(0., 2., 4.), thrust_curve_vac_n=(0., 4000., 0.))
    time_s = np.linspace(0., 4., 401)
    result = propulsion(stage, time_s, 0.)
    assert_allclose(result.thrust_n[[0, 100, 200, 300, 400]], [0., 2000., 4000., 2000., 0.])
    assert_allclose(np.trapezoid(result.mass_flow_kg_s, time_s), 8000./(200.*G0_M_S2), rtol=1e-12)
    assert stage.nominal_cutoff_s == 4.


@pytest.mark.parametrize("overrides", [
    {"dry_mass_kg": -1.}, {"isp_vac_s": 0.}, {"mass_flow_kg_s": np.nan},
    {"mass_flow_kg_s": None}, {"cutoff_time_s": -1.}, {"separation_delay_s": -1.},
    {"inertia_per_mass_m2": (1., 1., 10.)},
    {"thrust_curve_time_s": (0., 1.), "thrust_curve_vac_n": (1., 2.)},
])
def test_invalid_stage(overrides: dict[str, object]) -> None:
    """Reject invalid SI stage inputs and incompatible propulsion modes."""
    values = dict(name="invalid", dry_mass_kg=10., propellant_mass_kg=10., isp_vac_s=200., nozzle_exit_area_m2=0., mass_flow_kg_s=1.)
    values.update(overrides)
    with pytest.raises(ValueError):
        StageConfig(**values)


def test_preset_mass_ranges_and_launch_thrust() -> None:
    """Check class-level plausibility, without claiming achieved trajectory performance."""
    assert len(PRESETS) == 3
    masses_kg = [vehicle.initial_mass_kg for vehicle in PRESETS.values()]
    assert 2000. < masses_kg[0] < 4000.
    assert 12000. < masses_kg[1] < 14000.
    assert 500000. < masses_kg[2] < 600000.
    for vehicle in PRESETS.values():
        assert len(vehicle.stages) == 2
        thrust_n = float(propulsion(vehicle.stages[0], 0., 101325.).thrust_n)
        assert thrust_n > vehicle.initial_mass_kg * G0_M_S2
        assert all(stage.nominal_cutoff_s > 0. for stage in vehicle.stages)
    with pytest.raises(ValueError):
        VehicleConfig("empty", (), 0.)
