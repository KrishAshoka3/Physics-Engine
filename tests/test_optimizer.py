"""Mass allocation, actual structural constraints, budgets, and feasible search verification."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.analysis import DesignPoint, OptimizationConfig, evaluate_candidate, optimize_trajectory, sized_vehicle
from tests.analysis_cases import short_launch
from tests.guidance_cases import upper_vehicle


def test_fuel_allocation_preserves_total_and_engine_thrust() -> None:
    """Sizing conserves total fuel [kg] and fixed engine flow [kg/s], with explicit dry scaling."""
    vehicle, _, _ = short_launch()
    design = DesignPoint(.4, .05, 4.)
    sized = sized_vehicle(vehicle, design)
    assert_allclose(sum(stage.propellant_mass_kg for stage in sized.stages), 3.)
    assert_allclose([stage.propellant_mass_kg for stage in sized.stages], [1.2, 1.8])
    assert_allclose([stage.dry_mass_kg for stage in sized.stages], [9., 14.4])
    assert [s.mass_flow_kg_s for s in sized.stages] == [s.mass_flow_kg_s for s in vehicle.stages]
    fixed = sized_vehicle(vehicle, design, scale_dry_mass=False)
    assert [s.dry_mass_kg for s in fixed.stages] == [15., 8.]


@pytest.mark.parametrize("method", ["SLSQP", "differential_evolution"])
def test_real_trajectory_optimization_and_verified_constraints(method: str) -> None:
    """A feasible best candidate must improve actual apogee and satisfy independent load checks."""
    vehicle, simulation, guidance = short_launch()
    settings = OptimizationConfig(method=method, stage1_fraction_bounds=(.35, .8),
                                   pitch_kick_bounds_rad=(0., .1), max_q_pa=5000., max_axial_g=5.,
                                   max_iterations=5, population_size=3, max_evaluations=35, seed=19)
    result = optimize_trajectory(vehicle, simulation, guidance, settings)
    assert result.candidate.feasible
    assert result.candidate.metrics.max_q_pa <= settings.max_q_pa
    assert result.candidate.metrics.max_abs_axial_g <= settings.max_axial_g
    assert result.candidate.objective_value >= result.baseline.objective_value
    assert result.improved
    assert len(result.evaluations) <= 35
    assert result.configuration["verification_simulation"]["max_step_s"] == simulation.max_step_s/2.


def test_infeasible_search_and_horizon_are_not_success() -> None:
    """Insufficient propagation and impossible structural limits cannot produce a feasible optimum."""
    vehicle, simulation, guidance = short_launch()
    settings = OptimizationConfig(max_q_pa=.001, max_axial_g=.001, max_iterations=1, max_evaluations=3)
    result = optimize_trajectory(vehicle, simulation, guidance, settings)
    assert not result.candidate.feasible and not result.improved
    assert "infeasible" in result.message
    short = evaluate_candidate(vehicle, replace(simulation, duration_s=.1), guidance,
                                OptimizationConfig(), DesignPoint(2./3., .03, 3.))
    assert not short.feasible and np.isnan(short.metrics.apogee_altitude_m)
    with pytest.raises(ValueError, match="orbital"):
        optimize_trajectory(vehicle, simulation, guidance, OptimizationConfig(objective="payload_to_orbit"))
    outside = evaluate_candidate(vehicle, simulation, guidance,
                                  OptimizationConfig(stage1_fraction_bounds=(.4, .6)), DesignPoint(.7, .05, 3.))
    assert not outside.feasible
    assert np.all(outside.constraint_margins >= 0.)  # Only the design bound is violated.


def test_payload_optimization_requires_actual_final_stage_insertion() -> None:
    """Payload search must independently achieve orbital insertion, not merely increase launch mass."""
    from src.guidance import GuidanceProgram, OrbitTarget, SteeringConfig
    from src.physics import SimulationConfig, StageConfig, VehicleConfig
    booster = StageConfig("brief booster", 1., .3, 340., 0., mass_flow_kg_s=1., reference_area_m2=0.)
    vehicle = VehicleConfig("airborne sizing fixture", (booster, upper_vehicle().stages[0]), 10.)
    simulation = SimulationConfig(duration_s=100., atmosphere_enabled=False, rotating_earth=False, j2_enabled=False,
                                  launch_altitude_m=196000., initial_velocity_enu_m_s=(7400., 0., 200.), max_step_s=3.)
    guidance = GuidanceProgram(SteeringConfig(rail_length_m=0., target=OrbitTarget()))
    settings = OptimizationConfig(objective="payload_to_orbit", stage1_fraction_bounds=(.0008, .0012),
                                   pitch_kick_bounds_rad=(.01, .1), payload_bounds_kg=(5., 20.),
                                   max_q_pa=100., max_axial_g=5., max_iterations=3, max_evaluations=16)
    result = optimize_trajectory(vehicle, simulation, guidance, settings)
    assert result.candidate.feasible and result.candidate.metrics.insertion_achieved
    assert result.candidate.design.payload_mass_kg > 10.
    assert result.improved
    unfinished = evaluate_candidate(vehicle, replace(simulation, duration_s=.4), guidance, settings,
                                     DesignPoint(.001, .05, 10.))
    assert not unfinished.feasible
    assert np.isnan(unfinished.metrics.insertion_radius_error_m)
