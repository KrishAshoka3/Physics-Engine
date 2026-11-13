"""Acceptance checks for the in-progress Phase 4 benchmark and export APIs."""

from dataclasses import replace
import json

import numpy as np
import pytest

from src.analysis.validation import BenchmarkResult, analytical_benchmarks, openrocket_benchmarks
from src.physics import SimulationConfig, StageConfig, VehicleConfig, simulate
from src.physics.constants import G0_M_S2
from src.ui.mission import MissionSettings, build_mission, mission_export, telemetry_frame


def test_analytical_acceptance() -> None:
    """Require all nine independently specified analytical tolerances to pass."""
    rows = analytical_benchmarks()
    assert len(rows) == 9
    assert all(row.status == "PASS" for row in rows)


def test_openrocket_reduced_order_acceptance() -> None:
    """Require archived-reference apogee and speed errors below four percent."""
    rows, result, reference = openrocket_benchmarks()
    assert all(row.status == "PASS" and row.relative_error < .04 for row in rows)
    assert len(reference["sha256"]) == 64
    assert {"MECO", "stage_1_separation", "SES-1", "SECO"} <= {event.name for event in result.events}


def test_contextual_agreement_is_not_validation() -> None:
    """Even exact schedule agreement cannot be labeled observational validation."""
    row = BenchmarkResult("schedule", "contextual", 100., 100., "s", .04, "reference", "unmatched")
    assert row.status == "CONTEXT ONLY"
    assert replace(row, value=np.nan).status == "UNAVAILABLE"
    assert replace(row, category="analytical", value=105.).status == "FAIL"


def test_expended_delta_v_uses_event_left_limits() -> None:
    """Thrust quadrature converges to ideal delta-V without separation mass impulses."""
    vehicle = VehicleConfig("quadrature", (
        StageConfig("first", 10., 50., 300., 0., mass_flow_kg_s=10., coast_after_separation_s=2.),
        StageConfig("second", 5., 20., 320., 0., mass_flow_kg_s=10.)), 5.)
    config = SimulationConfig(duration_s=12., max_step_s=.02, atmosphere_enabled=False,
                              rotating_earth=False, j2_enabled=False, mu_m3_s2=0., launch_altitude_m=100.)
    result = simulate(vehicle, config)
    expected_m_s = G0_M_S2*(300.*np.log(90./40.)+320.*np.log(30./10.))
    assert result.telemetry["delta_v_expended_m_s"][-1] == pytest.approx(expected_m_s, rel=2e-5)
    assert np.all(result.telemetry["drag_n"] == 0.)
    coast = (result.time_s > 5.) & (result.time_s < 7.)
    assert np.ptp(result.telemetry["delta_v_expended_m_s"][coast]) < 1e-9
    frame = telemetry_frame(result)
    assert len(frame) == len(result.time_s)
    assert "position_eci_m_2" in frame and "state_7" in frame
    assert frame.events.str.contains("stage_1_separation").any()


def test_mission_configuration_export() -> None:
    """Export SI configuration faithfully and preserve immutable preset defaults."""
    settings = MissionSettings(thrust_scale=1.1, tvc=True)
    vehicle, config, guidance = build_mission(settings)
    document = json.loads(mission_export(settings))
    assert document["vehicle"]["name"] == vehicle.name
    assert document["simulation"]["dof"] == config.dof == 6
    assert guidance.steering.target is None
    with pytest.raises(ValueError):
        build_mission(replace(settings, thrust_scale=-1.))
