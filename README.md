# Aerospace Trajectory, Guidance & Mission Analysis Suite

Phase 1 provides atmospheric, aerodynamic, propulsion, vehicle, gravity/frame,
and event-driven 3DoF/6DoF simulation modules. Guidance and the UI are future phases.

From PowerShell in the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m tests.benchmark_physics
```

See [model documentation and usage](docs/physics_models.md),
[verification evidence](docs/phase1_audit.md), and [roadmap](ROADMAP.md).
Presets and upper-atmosphere/aerodynamic surrogates are engineering estimates;
analytical verification does not establish flight-data validation.
