# Phase 2 guidance and thrust vector control

This phase adds a sampled flight computer to the Phase 1 propagator, supporting
ideal 3DoF direction tracking and physical 6DoF pitch/yaw TVC. Numerical
verification establishes behavior under these models; it does not validate
a flight controller or reconstruct manufacturer guidance.

## Public API

```python
from src.guidance import AutopilotConfig, GuidanceProgram, OrbitTarget, SteeringConfig
from src.physics import PRESETS, SimulationConfig, simulate

vehicle = PRESETS["Small-Lift Orbital Vehicle (Electron Class)"]
program = GuidanceProgram(
    SteeringConfig(target=OrbitTarget(altitude_m=200_000.)),
    AutopilotConfig(mode="PID"),  # "LQR" is also supported
)
ideal = simulate(vehicle, SimulationConfig(dof=3, duration_s=460.), guidance=program)
controlled = simulate(vehicle, SimulationConfig(dof=6, duration_s=460.), guidance=program)
insertions = [row for row in controlled.guidance_records if row["phase"] == "insertion"]
assert insertions[-1]["achieved"]  # inspect actual errors, not just planner feasibility
```

`OrbitTarget(200_000., 500_000.)` requests insertion at the perigee of a 200 by
500 km ellipse. Target altitudes use spherical radius minus WGS84 equatorial
radius, consistent with orbital telemetry; they are not geodetic height.
Azimuth sets the approximate orbital plane; inclination is not targeted.
`target=None` selects rail/kick/gravity-turn guidance, even for an orbital preset.

Set `rail_length_m=0.` for airborne initialization. Guided 6DoF requires every
stage to have an aft axial `thrust_offset_body_m=(-L, 0, 0)` with L > 0 m.
Guidance cannot be mixed with legacy `thrust_direction` or `control` callbacks.
Omitting `guidance` preserves the Phase 1 default vertical 3DoF direction.
Every simulation creates independent controller/planner memory.

## Interfaces, frames, and states

`src/physics/flight.py` owns the `FlightProgram` / `FlightSession` protocols,
`NavigationState`, and `FlightCommand`. Physics never imports the guidance
package. A future estimator can wrap a flight session with estimated navigation;
this phase supplies simulation truth, with no sensor noise, latency, or filter.

Navigation position [m], inertial velocity [m/s], and air-relative velocity [m/s]
are always ECI vectors. Air velocity subtracts Earth corotation and all three
components of altitude-dependent geographic ENU wind. Matrices map body,
current geographic ENU, or launch ENU axes into ECI. Time is mission elapsed
seconds; stage indices and flags are dimensionless. See [physics conventions](physics_models.md).

| Mode | State columns | Units |
| --- | --- | --- |
| 3DoF | position 0:3, velocity 3:6, mass 6, active fuel 7 | m, m/s, kg, kg |
| 6DoF | above plus quaternion 8:12, inertial body rate 12:15 | dimensionless, rad/s |
| Guided 6DoF | above plus actual pitch/yaw gimbals 15:17 | rad |

The Hamilton scalar-first quaternion `[w,x,y,z]` maps body to propagation frame.
Body rate is inertial angular velocity in body axes. ECI uses
q_dot = 0.5 q tensor-product [0,omega]; rotating ENU subtracts frame angular
velocity for q_dot. Matrix construction normalizes q; integrated states are
not clipped or renormalized after steps. Norm drift is tested. Attitude errors
use the shortest arc and are invariant to q versus -q. Two-axis TVC controls
nose direction; roll is unconstrained.

## Rail, kick, and gravity turn

Defaults are a 15 m vertical rail, eastward azimuth pi/2 rad, a 5 s post-release
hold, and a 5 degree pitch-kick over 10 s. All angles are radians internally.
Rail translation is constrained along an Earth-fixed axis; 6DoF attitude stays
aligned with the correct inertial Earth rotation rate. Release is the ascending
terminal root `(position_ecef-origin) dot axis - rail_length = 0` [m], with no
state impulse. Both propagation frames include consistent transport terms.
A supplied rail state must start at the origin with along-rail velocity and
matching attitude/rate. This is a rigid frictionless rail, without flexure or a
launch-pad support latch; an unsupported launch may impact immediately.

The hold uses the rail axis; the kick uses local geographic elevation and the
configured azimuth. `pitch_profile="linear"` uses a bounded linear ramp.
The exponential option uses u=(1-exp(-4s))/(1-exp(-4)), followed by 3u^2-2u^3
for normalized maneuver time s in [0,1], giving zero-slope endpoints.
Gravity-turn commands align with air-relative velocity; below 1e-6 m/s the
unit rail axis is used. Ideal mode has zero incidence in this phase. Actual
6DoF incidence depends on control and disturbances. The handoff to velocity
following can introduce a finite commanded-direction change.

## Closed-loop orbital steering

Upper-stage guidance starts at `orbital_stage_index` (default 1). It implements
numerical linear-tangent shooting inspired by powered explicit guidance, not
the Shuttle/SLS flight PEG algorithm. Direction uses current ECI radial and
prograde tangential axes, with

`tan(pitch(t)) = tan(pitch0) + (tan(pitch1)-tan(pitch0))*t/time_to_go`.

`scipy.optimize.least_squares` solves pitch endpoints [rad] and remaining burn
[s] against insertion radius [m], radial velocity zero [m/s], and vis-viva
tangential speed [m/s]. The central-gravity vacuum predictor integrates
dr/dt=vr, dvr/dt=a*sin(pitch)+vt^2/r-mu/r^2,
dvt/dt=a*cos(pitch)-vr*vt/r, dm/dt=-flow. Acceleration is vacuum thrust/mass.
Flow follows the active stage's constant or piecewise-linear curve. Pitch
bounds are +/-1.45 rad; burn is bounded by the exact fuel integral and scheduled
cutoff. Residual scales are 10000 m / 1000 m/s / 1000 m/s; the solve is limited
to 45 function evaluations. Predictor DOP853 uses rtol=2e-8 and
atol=[0.001 m, 1e-5 m/s, 1e-5 m/s, 1e-7 kg].

Planning defaults to 5 s. Ideal-mode updates shorten near insertion to 20% of
time-to-go, with a 0.1 s minimum. In 6DoF, full pitch replanning stays on the
5 s cadence. During the final `terminal_guidance_s` (default 5 s), the existing
tangent law is preserved while a scalar speed root trims cutoff. These
speed-only updates use the shorter cadence, avoiding ill-conditioned
last-second turns faster than the attitude controller can follow.

Feasibility requires radius error <=100 m and both velocity errors <=1 m/s by
default. Failed predictions are recorded. Without a feasible plan, gravity turn
continues until the physical fuel/time cutoff. A later failed solve retains
the previous feasible plan. Guided cutoff is a terminal `solve_ivp` clock event
that preserves residual fuel. Actual insertion errors are measured separately
at cutoff and marked `achieved=True/False`.

The predictor omits atmosphere, J2, future stage/fairing jumps, and attitude lag.
Replanning responds to trajectory deviations and mass changes, but does not
guarantee convergence or feasibility. Multi-burn mission optimization is outside
this phase, as are inclination targeting and unpowered attitude control.

## PID/LQR and physical actuators

Control updates at accepted 0.1 s clocks and at phase/stage/planning events.
Minimal nose-alignment error supplies pitch/yaw feedback. Desired-direction
rate feed-forward uses a 0.5 s low-pass filter to suppress derivative kicks
from discrete replanning. PID defaults are 1.2 rad/s natural frequency, 0.9
damping, and integral gain 0.08 1/s^3. Integrated error is bounded to
+/-0.2 rad*s; accumulation stops under travel saturation or servo lag exceeding
0.1 degree from the prior target. Rail/stage transitions reset memory.

LQR solves a continuous Riccati equation for a normalized angular double
integrator with angle/rate weights 2.0736/1.7856 and effort weight 1. Angular
acceleration feedback maps to moments using current inertia, with gyroscopic
and inertia-rate compensation. This is a local, inertia-scheduled design,
not a global nonlinear stability proof.

An aft nozzle at [-L,0,0] m produces force
F*[cos(p)cos(y), cos(p)sin(y), -sin(p)] N. Its offset cross force is the only
TVC torque added to the rigid-body equations. A zero nozzle arm is rejected.
Travel is +/-5 degrees per axis, not a 5 degree resultant cone. Actual servo
states obey d(gimbal)/dt=clip((setpoint-gimbal)/0.08 s, +/-20 degrees/s).
Setpoints are bounded; actual states integrate without post-step clipping.
No thrust means no TVC authority. Final-stage coast skips idle controller clocks
while preserving servo dynamics and physical events.

Preset nozzle offsets assume -0.4 times the active stack length:
Black Brant-class -5.2/-3.48 m, Electron-class -7.2/-2.4 m, Falcon-class
-28/-6.4 m. These are provisional geometry, not published mass-center data.
Existing cylinder inertia assumptions remain. Shear verification uses a separate
explicit -0.5 m aerodynamic offset and -2 m nozzle arm.

## Timing and diagnostics

Controller memory never changes inside adaptive RHS calls. Accepted samples
return pure immutable direction laws and held actuator setpoints. Laws may
evaluate current navigation between clocks; only actuator setpoints are held
constant. The solver terminates/restarts at clocks and physical events.
Coincident-root priority is impact, fuel/time/guided cutoff, stage transition,
fairing, rail, then flight update. Remaining roots are handled on restart.
Output epochs within 32 machine epsilons of the epoch scale are coalesced into
a right-hand sample; exact event roots remain in the physical event ledger.

Telemetry adds `command_axis_eci` (dimensionless), `attitude_error_rad` (nose
error, zero by definition in ideal mode), `rail_attached` (0/1), and in guided
6DoF `gimbal_rad` / `gimbal_target_rad` (pitch/yaw columns). `guidance_records`
stores sample time [s], phases, saturation, planner feasibility, pitch/rate
[rad,rad/s], burn [s], terminal pitch-freeze flags, and actual insertion errors
[m,m/s,m/s]. Control-clock updates do not clutter the physical event ledger.

## References and evidence

- [NASA SLS ascent flight control](https://ntrs.nasa.gov/api/citations/20140008731/downloads/20140008731.pdf): guidance/control architecture background; not a source for these gains.
- [NASA unified powered flight guidance](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19740004402.pdf): powered-guidance background; this implementation uses the numerical approximation described above.
- [SciPy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) and [Riccati solver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_continuous_are.html): numerical solver definitions.

The [Phase 2 audit](phase2_audit.md) records acceptance criteria, insertion,
wind-shear tracking, convergence, profiling, and remaining limitations.
