# Aerospace Trajectory, Guidance & Mission Analysis Suite

Phase 1 provides atmospheric, aerodynamic, propulsion, vehicle, gravity/frame,
and event-driven 3DoF/6DoF simulation modules. Phase 2 adds rail/pitch/gravity-turn
guidance, orbital linear-tangent steering, and PID/LQR TVC with physical gimbal
travel and rate limits. Phase 3 adds reproducible parallel launch dispersions,
covariance ellipses, regression/Sobol sensitivity, and constrained fuel-split,
pitch, and payload optimization. Reporting and the UI remain future phases.

From PowerShell in the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m tests.benchmark_physics
.\.venv\Scripts\python.exe -m tests.benchmark_guidance
.\.venv\Scripts\python.exe -m tests.benchmark_analysis
```

```python
from src.guidance import GuidanceProgram, OrbitTarget, SteeringConfig
from src.physics import PRESETS, SimulationConfig, simulate

program = GuidanceProgram(SteeringConfig(target=OrbitTarget(200_000.)))
result = simulate(
    PRESETS["Small-Lift Orbital Vehicle (Electron Class)"],
    SimulationConfig(dof=6, duration_s=460.),  # dof=3 selects ideal tracking
    guidance=program,
)
```

See [physics models](docs/physics_models.md), [guidance usage](docs/guidance_models.md),
[analysis usage and assumptions](docs/analysis_models.md),
[Phase 1 evidence](docs/phase1_audit.md), [Phase 2 evidence](docs/phase2_audit.md),
[Phase 3 evidence](docs/phase3_audit.md), and [roadmap](ROADMAP.md).
Presets and upper-atmosphere/aerodynamic surrogates are engineering estimates;
analytical verification does not establish flight-data validation.
