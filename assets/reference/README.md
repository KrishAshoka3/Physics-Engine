# Reference provenance

`openrocket_two_stage_24_12.ork` is the unmodified **Two stage high power rocket**
example from OpenRocket's `release-24.12` tag:
https://github.com/openrocket/openrocket/blob/release-24.12/core/src/main/resources/datafiles/examples/Two%20stage%20high%20power%20rocket.ork

Retrieved 2026-11-12. Copyright OpenRocket contributors; upstream GPL license
is included in `OpenRocket-LICENSE.txt`. No Java runtime is required to read the
saved simulation data. The suite does **not** claim to have rerun OpenRocket.
The report records the archive SHA-256 and uses Simulation 1, Sustainer branch.

The comparison is a reduced-order two-stage integration benchmark. It starts
from the recorded liftoff state at 0.067 s, adopts stored thrust histories,
derives equivalent constant Isp from impulse / recorded propellant consumption,
and uses the median stored pre-apogee drag coefficient of each stage (0.880 and
0.586). The last pre-separation mass sample approximates the left limit; stored
values are rounded. Discarded mass is inferred from that same branch, not from
the separately reported booster branch. These shared inputs are not independent
aerodynamic validation. No apogee or velocity outputs are used to tune inputs.
Wind, changing attitude, pad hold, variable Cd, and parachutes are not reproduced;
only ascent/apogee and peak ascent speed are compared. The example is a small
high-power model, not a flight-validated Black Brant model.

Public orbital references (retrieved 2026-11-12):

- NASA Commercial Crew press kit, Crew-2 approximate schedule:
  https://www.nasa.gov/commercial-crew-program-press-kit/
  MECO 156 s, stage separation 159 s, SES-1 167 s, SECO-1 527 s.
- NASA TROPICS Electron schedule, 25 May 2023:
  https://www.nasa.gov/blogs/smallsatellites/2023/05/25/rocket-lab-tropics-timeline-of-events-for-todays-launch/
  MECO 149 s, separation 153 s, SES-1 156 s, SECO 567 s.

These are published schedules, not measured telemetry. Unmatched payload,
throttle, launch site and target orbit prevent observational validation. The
dashboard/report show actual discrepancies and never turn contextual agreement
into a validation pass. No reference page or press-kit artwork is redistributed.
