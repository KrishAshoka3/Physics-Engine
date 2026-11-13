# Phase 4 / 5 intermediate milestone

Work paused on 2026-11-13 at the user's request to push all current changes and
continue UI later. Neither phase is complete.

## Implemented

- `src/analysis/validation.py`: executable rocket-equation, burnout-time,
  eccentric-orbit energy/vis-viva and US76 pressure checks; explicit evidence
  classes, strict relative-error thresholds and unavailable/failure statuses.
- Archived OpenRocket 24.12 two-stage ascent comparison with original ORK,
  upstream license, SHA-256 reporting and documented reduced-order mapping.
  This reads saved OpenRocket output; it does not rerun OpenRocket.
- Actual solver-event comparisons against NASA's approximate Electron TROPICS
  and Falcon 9 Crew-2 schedules. These are context, not measured telemetry or
  observational validation; discrepancies are not tuned away.
- Solver drag force [N] and sampled expended thrust delta-V [m/s]. Delta-V uses
  trapezoidal integration of thrust/mass with pre-event left limits. It is a
  step-dependent diagnostic, not net velocity gain or ideal vacuum delta-V.
- Shared SI mission configuration, full telemetry DataFrame/CSV-ready export,
  replay configuration JSON, uncertainty scaling and parametric-study helpers.
- Playwright, imageio, imageio-ffmpeg and pypdf requirements; Chromium installed
  locally. Browser caches and temporary artifacts remain outside Git.

## Verification evidence

Commands use `.venv/Scripts/python.exe` from the repository root:

```text
python -m pytest -q
python -c "from src.analysis.validation import public_milestones; print([r.row() for r in public_milestones()])"
git diff --check
```

New acceptance tests cover all nine analytical rows, both OpenRocket metrics,
event-aware thrust quadrature across stage separation and coast, contextual
status semantics, full telemetry exports, and mission configuration validation.

Full regression result: **156 passed in 142.84 s**. `git diff --check` passed.
Installed browser/report dependencies: Playwright 1.65.1, imageio 2.38.1,
imageio-ffmpeg 0.8.1, pypdf 7.0.1. The Chromium installation succeeded;
no browser demo has been executed.

Measured analytical relative errors: rocket delta-V 2.77e-10; worst eccentric
orbit energy 2.28e-15; worst vis-viva speed 1.07e-15; US76 pressure at 86 km
1.24e-6. These meet their predeclared thresholds. OpenRocket apogee: 682.511 m
versus 674.979 m (1.116%); maximum speed: 159.565 m/s versus 158.799 m/s
(0.483%). Both meet the 4% reduced-order comparison target. The shared reference
thrust/mass and median drag coefficients prevent an independent aerodynamic
validation claim. See `assets/reference/README.md` for assumptions and sources.

Public schedule discrepancies: Electron MECO/separation/SES-1/SECO are
15.47% / 17.68% / 17.34% / 27.58%; Falcon 9 corresponding differences are
6.91% / 8.66% / 11.24% / 16.62%. All are **CONTEXT ONLY**, with unmatched
payload, throttle history, site and orbit. A universal sub-4% flight-validation
claim is unsupported. Percent relative error is not statistical variance.

## Remaining work

The report generator currently contains only a docstring explicitly identifying
it as unfinished. No `Engineering_VV_Report.pdf`, `app.py`, demo recording script,
WebM or GIF has been created. Report layout/render inspection, dashboard browser
testing, final performance audit, portfolio README and phase-completion gates
remain for a later authorized continuation. Do not use the original requested
phase-completion commit message for this partial milestone.
