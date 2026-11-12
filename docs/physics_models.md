# Phase 1 core physics: conventions, models, and limits

This implementation supplies reusable numerical kernels and a hybrid event-driven propagator. It is analytically verified within the cases in `tests/`; it is not a flight-qualified or empirically validated vehicle model. Phase 2 guidance and TVC are available through the optional `guidance` argument; see [guidance models and usage](guidance_models.md) for the 17-state guided 6DoF schema and provisional preset nozzle offsets.

## Usage

Run from the repository root using the installed `.venv` interpreter:

```python
from src.physics import PRESETS, SimulationConfig, simulate

vehicle = PRESETS["Two-Stage Sounding Rocket (Black Brant IX Class)"]
result = simulate(vehicle, SimulationConfig(duration_s=700.0))
print(result.termination)
print([(event.name, event.time_s) for event in result.events])
print(result.telemetry["altitude_m"].max())
```

The default 3DoF direction is local geodetic up throughout flight. It is an intentionally simple force-direction assumption, not gravity-turn or orbital guidance. Supply `thrust_direction(time_s, state)` returning a unit vector in the selected propagation frame to prescribe another attitude. The same axis resolves aerodynamic incidence. A callback must be pure, since adaptive integration evaluates trial states out of time order. Use `dof=6` for integrated attitude and `control(time_s, state) -> ControlInput` for a body-frame thrust direction and applied body torque. The 6DoF mode does not accept a 3DoF direction callback.

`engines_enabled=False` starts a fully fueled, unpowered coast. `initial_state()` constructs a state from launch coordinates and Earth-relative ENU velocity; `state0=` can override position, velocity, and attitude for orbital tests. Its masses must match the supplied vehicle. Change the vehicle configuration to model a different starting mass.

## Frames and state schema

All numerical inputs and outputs use SI units. Angles and angular rates use radians. Flags, indices, Mach, coefficients, eccentricity, TWR, axial G-load, quaternion components, and rotation matrices are dimensionless. Names, selectors, and labels have no physical unit.

| Columns | Quantity | Units |
| --- | --- | --- |
| 0:3 | Propagation-frame position | m |
| 3:6 | Propagation-frame coordinate velocity | m/s |
| 6 | Total attached mass, including fuel, payload, and retained fairing | kg |
| 7 | Current-stage remaining propellant | kg |
| 8:12, 6DoF only | Hamilton quaternion, scalar first `[w,x,y,z]`, body to propagation frame | dimensionless |
| 12:15, 6DoF only | Body angular velocity **relative to inertial space**, expressed in body axes | rad/s |

Body +x points toward the nose; nominal thrust is +x. Initial azimuth is measured east of north and elevation above the local horizon. The initial roll basis is constructed in ENU and then transformed, so it represents the same physical orientation in either mode.

- **ECI:** geocentric inertial position and velocity. ECI and ECEF axes coincide at elapsed mission time zero. Earth-fixed vectors rotate about +z at the nominal Earth rate when `rotating_earth=True`. There are no fictitious accelerations in ECI. Initial velocity includes `omega cross r`.
- **ENU:** a rotating, launch-site-fixed orthonormal basis anchored at the configured WGS84 location. Position is relative to the launch-site origin. Every gravity evaluation reconstructs the full geocentric position. This is a curved-Earth propagation expressed in a fixed ENU basis, not a flat-Earth height approximation.
- ENU acceleration includes `-2 omega cross v_relative - omega cross (omega cross r_geocentric)` before projecting into ENU. The origin's acceleration is included through the **geocentric** centrifugal position.
- Wind is a 3D piecewise-linear profile in the **current location's** geographic ENU basis, relative to Earth. ECI air-relative velocity subtracts both atmospheric corotation and wind. Endpoint winds are held constant outside the supplied height range.
- The ECI epoch is a modeling convention, not an IERS/GCRF realization. No UTC/UT1, precession, nutation, polar motion, geoid undulation, or terrain model is included. Atmospheric geometric height uses WGS84 ellipsoid height as its vertical datum.

## Atmosphere

The seven base geopotential heights are 0, 11, 20, 32, 47, 51, and 71 km. Hydrostatic pressure is derived recursively from sea level, using the prescribed lapse rates and the COESA gas constant. Geometric/geopotential conversion uses the atmosphere's 6,356,766 m reference radius, deliberately separate from the WGS84 ellipsoid. The geometric 86 km boundary is approximately 84.852 km geopotential.

Temperature returned to callers is **kinetic temperature**. The small 80--86 km molecular-weight correction uses shape-preserving interpolation of US76 Table 8. Density and pressure retain their molecular-scale equations; the exposed `gas_constant_j_kg_k` makes `P = rho R T` consistent. Sound speed is `sqrt(gamma R T)` with gamma 1.4.

Above 86 km, with geometric increment `x` [m], density follows:

`rho = rho86 exp[-x/H_inf - (k86 - 1/H_inf) L (1 - exp(-x/L))]`

Here `H_inf = 50000 m`, `L = 60000 m`, and `k86 = -d(log rho)/dz` at the boundary. A relaxed exponential temperature approaches 1000 K and matches the lower-side temperature derivative. The tiny boundary molecular-weight derivative is also relaxed continuously. Pressure follows the ideal-gas relation. Temperature, pressure, density, and sound speed are C1 across 86 km, and remain positive and finite through 500 km.

This **upper-atmosphere surrogate is not the full US76 species-diffusion model**, is not hydrostatically self-consistent with local gravity everywhere, and has no solar/geomagnetic/time dependence. Its density scale parameters are engineering assumptions. Continuation above 500 km is mathematical only; orbital lifetime and rarefied-flow predictions require a validated thermosphere and aerodynamic model. Below sea level the first layer continues to -5000 m for below-datum sites and integrator trial states. Inputs below that limit are rejected. Sound speed in near-vacuum is a diagnostic, not a claim that continuum acoustics applies.

## Aerodynamics and propulsion

`coefficients()` broadcasts Mach and angle of attack. A smooth transonic onset over roughly Mach 0.85--1.25 adds a wave-drag peak, followed by supersonic decay to an axial hypersonic asymptote. Lift blends from a small-angle slope to a signed Newtonian `2 sin(alpha) |sin(alpha)| cos(alpha)` form; incidence drag uses `2 sin^2(alpha)`. Both coefficients depend on Mach and alpha. Drag is even in signed alpha and lift is odd. Force resolution uses the plane containing the nose and relative velocity; drag opposes flow and lift is perpendicular to it. Zero speed returns zero force without a normalization singularity. `Q = rho |v_relative|^2 / 2` uses all three wind components.

These are configurable, smooth **engineering coefficient curves**, not measured Black Brant, Electron, or Falcon aerodynamic databases. They omit Reynolds-number effects, heating, ablation, rarefaction, side-force asymmetries, buffet, and aeroelasticity. Large-angle behavior is a mathematical continuation. Replace/calibrate them before vehicle-specific conclusions.

Each stage selects either positive constant mass flow or arbitrary nonnegative piecewise-linear **vacuum** thrust samples, starting at local ignition time zero. For curves, consumption is `F_vac / (Isp_vac g0)`. `cutoff_time_s` can terminate a burn before depletion; otherwise the scheduled limit is fuel/flow or the curve endpoint. Depletion remains an independent terminal root.

Powered thrust is `F_vac - P_ambient A_exit`. A negative value is intentionally not hidden by clipping: it indicates an overexpanded/no-longer-valid nozzle approximation. An off engine, or exactly zero curve flow, produces no thrust and consumes no fuel. Engine on/off is controlled by event mode, never by a timestep comparison in the force model. Thrust curves continue at their endpoint values for solver trial evaluations; event functions enforce the burn domain.

## Gravity and rigid-body dynamics

Gravity consists of the central inverse-square term plus the degree-2 zonal J2 term, using WGS84 geometry and constants. It does not add WGS84 normal gravity on top of J2. Set `j2_enabled=False` for two-body conservation tests. `mu_m3_s2=0` is available for force-isolation tests such as Tsiolkovsky.

6DoF integrates translation, quaternion kinematics, and Euler's rotational equation:

`I omega_dot = torque - omega cross (I omega) - I_dot omega`

The quaternion uses the angular velocity relative to the propagation frame; in rotating ENU, Earth rate is transformed into the body and subtracted. Inertial body rate, not Euler angles, is the rotational state. A normalized local quaternion copy is used to form rotation matrices; the integrated quaternion is not clipped or renormalized after every step. Norm drift is exposed in telemetry and tested.

Applied moments include the force at configured aerodynamic-center and nozzle offsets, optional linear body-rate damping, and the control callback's torque. The default offsets and damping are zero. Principal inertia is total attached mass times configurable moments-per-mass, defaulting to a uniform-cylinder stack approximation. The derivative uses the continuous fuel mass loss. **This assumes zero explicitly modeled exhaust angular-momentum flux.** Tank-driven CM migration, jet damping, slosh, flexible modes, and realistic exhaust angular momentum are not modeled. Real spinning rockets require validated inertia/CM/efflux data. At a discrete jettison, attitude and angular rate remain continuous; departing material carries away its share of angular momentum with no applied separation impulse. Stage dry mass is stage-only, while its inertia geometry and force offsets describe the currently attached stack.

## Event sequence and history

All physical transitions are `solve_ivp` terminal events, with appropriate crossing directions:

1. Stage ignition begins a smooth powered segment. Fuel [kg] crossing downward or a scheduled cutoff clock [s] ends it (MECO for the first stage, SECO for the second).
2. A terminal separation clock follows the configurable `separation_delay_s`. Separation removes the stage's dry mass **and any residual fuel**. Position, velocity, attitude, and body rate do not jump.
3. The next stage's terminal ignition clock follows `coast_after_separation_s`; the second-stage ignition is SES-1. Fuel is not consumed during either coast interval.
4. Fairing jettison is a separate absolute-time or ascending-geodetic-altitude terminal event. It subtracts its mass exactly once. Reference aerodynamic area/force offsets remain those of the current stage.
5. Descending ground intersection terminates propagation. Initial ascent from the surface is allowed; a launch without sufficient support can terminate at time zero. The final stage remains attached after cutoff.

Known clock times also bound solver trial intervals, but transitions are dispatched only from reported event roots. Zero-delay transitions restart at exactly the same epoch. When roots coincide, priority is impact, depletion/cutoff, stage transition, then fairing; remaining roots are detected on restart. The event ledger preserves pre/post states. Returned sample times are strictly increasing, and shared event rows use the post-transition mode/state. At the configured time horizon propagation stops; a further simultaneous zero-duration stage chain is not advanced beyond that horizon.

The solver samples adaptively with no default uniform output grid. If resampling for a later UI, interpolate each smooth segment separately; never interpolate mass across a jettison. `max_step_s` still matters for rapidly changing forces and multiple crossings. Event detection is root-based, but no finite-step integrator can guarantee discovery of arbitrarily many crossings in one step.

## Telemetry

`telemetry` contains geometric altitude, air-relative speed, Mach, Q, total incidence, net thrust, axial G-load, TWR, mass, remaining propellant, ECI position/velocity, and orbital diagnostics. Axial G is the nongravitational specific force along the nose divided by standard gravity (zero in free fall). TWR uses local gravitational acceleration magnitude, excluding fictitious terms; it is NaN in zero-gravity test cases.

Semi-major axis, eccentricity, and inclination are **two-body osculating elements** from ECI position and velocity. Instantaneous apogee and perigee subtract the WGS84 equatorial radius; these are not ellipsoidal geodetic altitudes or a propagated forecast under continuing thrust/J2. Hyperbolic/parabolic apogee is +infinity, radial inclination is NaN, and all orbital elements are NaN when mu is zero. Specific energy excludes the J2 potential and is conserved only in the unforced spherical-gravity benchmark.

## Presets and provenance

| Preset | Initial mass | Modeled payload | What is assumed |
| --- | ---: | ---: | --- |
| Black Brant IX class | 3308 kg | 650 kg | Heavy sounding mission; 350 kg booster dry, 760 kg booster fuel; 500 kg upper dry includes about 274 kg motor hardware plus 226 kg assumed vehicle/recovery structure; 1008 kg upper fuel; 40 kg retained fairing |
| Electron class | 12994 kg | 200 kg | 950/9250 kg first-stage dry/fuel, 250/2300 kg second-stage dry/fuel, 44 kg fairing; no kick stage or battery jettison |
| Falcon 9 class | 534870 kg | 15000 kg | 25600/395700 kg first-stage dry/fuel, 4000/92670 kg upper dry/fuel, 1900 kg fairing; no recovery fuel reserve or throttle schedule |

The sounding example is intended to represent an approximately 200 km **class** mission; the model's default vertical ascent reaches about 216 km. Added non-motor hardware masses are explicit engineering assumptions, not values claimed by NASA. The initial motor-only budget reached about 317 km and was replaced with a complete-stage budget rather than presented as a validated 200 km flight. Electron's roughly 7.6 km/s insertion is a mission target, not a result of the default vertical direction. All preset nozzle exit areas, mass allocations not explicitly anchored below, cylinder inertias, and aerodynamic coefficients are assumptions. No OpenRocket or flight-telemetry validation is claimed.

Reference anchors (accessed 2026-11-12):

- [NOAA/NASA/USAF, U.S. Standard Atmosphere 1976](https://ntrs.nasa.gov/citations/19770009539): seven-layer atmosphere, hydrostatic equations, constants, and Table 8 molecular-weight ratios; [NOAA copy](https://www.ngdc.noaa.gov/stp/space-weather/online-publications/miscellaneous/us-standard-atmosphere-1976/us-standard-atmosphere_st76-1562_noaa.pdf).
- [NGA, WGS84 defining parameters](https://earth-info.nga.mil/?action=wgs84&dir=wgs84): ellipsoid axes/flattening, gravitational parameter, Earth rotation rate. [NOAA gravity reference-field notes](https://www.ngs.noaa.gov/GEOID/tech2.html) provide the J2 magnitude cross-check.
- [NASA Sounding Rocket Program Handbook, Appendix A.4](https://ntrs.nasa.gov/api/citations/20230006855/downloads/NASA%20TP%2020230006855%202023.05.08.pdf): Terrier/Black Brant architecture and Black Brant motor duration, propellant, and loaded motor mass. The model converts approximately 2223 lb of fuel and 2827 lb loaded motor mass to about 1008/1282 kg; its vacuum Isp is an estimate.
- [Rocket Lab, Electron](https://rocketlabcorp.com/launch/electron/): approximately 13000 kg launch mass, 1.2 m diameter, 224 kN first-stage peak thrust, 311 s listed Isp, 25.8 kN upper thrust, and 343 s upper Isp. Treating the first-stage listed Isp as vacuum Isp and selecting exit area to approach 190 kN sea-level thrust are modeling assumptions.
- [SpaceX, Falcon 9](https://www.spacex.com/vehicles/falcon-9/): vehicle dimensions, nine first-stage engines, and 981 kN upper-stage thrust. Stage mass split, vacuum Isp, and nozzle areas are estimated; this preset is not a reconstruction of the published vehicle's exact mass or burn schedule.
- [SciPy `solve_ivp` reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html): terminal/directional events and event-detection limitations.
