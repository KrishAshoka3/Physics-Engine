"""Executable analytical verification and qualified external comparisons.

Notes
-----
All numeric comparisons carry units and an acceptance limit declared before
evaluation. Public mission schedules are contextual, not measured flight truth.
OpenRocket is a reduced-order cross-tool case with shared input data, not an
independent validation of aerodynamic coefficients or a full ORK importer.
"""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal
from zipfile import ZipFile
import hashlib
import xml.etree.ElementTree as ET

import numpy as np

from src.guidance import GuidanceProgram, OrbitTarget, SteeringConfig
from src.physics import (AeroConfig, PRESETS, SimulationConfig, SimulationResult,
                         StageConfig, VehicleConfig, atmosphere, simulate)
from src.physics.constants import EARTH_MU_M3_S2, G0_M_S2, WGS84_A_M
from .metrics import trajectory_metrics

ROOT = Path(__file__).resolve().parents[2]
OPENROCKET_SOURCE = "https://github.com/openrocket/openrocket/blob/release-24.12/core/src/main/resources/datafiles/examples/Two%20stage%20high%20power%20rocket.ork"
FALCON_SOURCE = "https://www.nasa.gov/commercial-crew-program-press-kit/"
ELECTRON_SOURCE = "https://www.nasa.gov/blogs/smallsatellites/2023/05/25/rocket-lab-tropics-timeline-of-events-for-todays-launch/"
US76_SOURCE = "https://ntrs.nasa.gov/citations/19770009539"


@dataclass(frozen=True)
class BenchmarkResult:
    """One comparison with explicit unit, relative tolerance, and evidence class.

    Attributes
    ----------
    value, reference : float
        Model and reference values in ``unit``; NaN means unavailable.
    tolerance_fraction : float
        Dimensionless strict relative acceptance limit.
    category : str
        analytical, cross-tool, or contextual; contextual rows never pass validation.
    source, notes : str
        Human-readable provenance and model compatibility qualifications.
    """

    name: str
    category: Literal["analytical", "cross-tool", "contextual"]
    value: float
    reference: float
    unit: str
    tolerance_fraction: float
    source: str
    notes: str

    @property
    def relative_error(self) -> float:
        """Return absolute relative error (dimensionless), NaN if undefined."""
        return abs((self.value-self.reference)/self.reference) if self.reference != 0. else np.nan

    @property
    def status(self) -> str:
        """Separate numerical agreement from unsupported observational validation."""
        if not np.isfinite(self.relative_error):
            return "UNAVAILABLE"
        if self.category == "contextual":
            return "CONTEXT ONLY"
        return "PASS" if self.relative_error < self.tolerance_fraction else "FAIL"

    def row(self) -> dict[str, object]:
        """Return a serializable row; percent denotes relative error, not variance."""
        return dict(asdict(self), error_percent=100.*self.relative_error, status=self.status)


def analytical_benchmarks() -> list[BenchmarkResult]:
    """Execute rocket-equation, eccentric-orbit, event, and US76 reference checks.

    Returns
    -------
    list of BenchmarkResult
        Fresh solver measurements with strict acceptance thresholds.

    Notes
    -----
    Vacuum test: 120 to 20 kg, Isp 300 s, 10 kg/s, no gravity/rotation/air.
    Orbit: equatorial 400 x 1200 km, one period, central gravity only. Vis-viva
    uses the specified semi-major axis, not one recomputed from each state.
    """
    vehicle = VehicleConfig("Analytical verification", (
        StageConfig("Motor", 10., 100., 300., 0., mass_flow_kg_s=10.),), 10.)
    config = SimulationConfig(duration_s=12., launch_altitude_m=100., atmosphere_enabled=False,
                              rotating_earth=False, mu_m3_s2=0., j2_enabled=False,
                              max_step_s=2., rtol=1e-9, velocity_atol_m_s=1e-10)
    result = simulate(vehicle, config)
    exact_m_s = 300.*G0_M_S2*np.log(6.)
    rows = [BenchmarkResult("Tsiolkovsky delta-V", "analytical",
                            float(np.linalg.norm(result.state[-1, 3:6]-result.state[0, 3:6])),
                            float(exact_m_s), "m/s", 1e-4, "Tsiolkovsky: Isp*g0*ln(m0/mf)",
                            "Zero drag, gravity and rotation; no separation during this burn."),
            BenchmarkResult("Burnout event time", "analytical", result.events[0].time_s,
                            10., "s", 1e-8, "Propellant mass / mass flow", "Root-located depletion event.")]
    radius_m = WGS84_A_M+400000.
    semi_major_m = WGS84_A_M+800000.
    speed_m_s = np.sqrt(EARTH_MU_M3_S2*(2./radius_m-1./semi_major_m))
    period_s = 2.*np.pi*np.sqrt(semi_major_m**3/EARTH_MU_M3_S2)
    orbit = simulate(vehicle, SimulationConfig(duration_s=float(period_s), max_step_s=60.,
        launch_altitude_m=400000., engines_enabled=False, atmosphere_enabled=False,
        rotating_earth=False, j2_enabled=False, initial_velocity_enu_m_s=(float(speed_m_s), 0., 0.)))
    radii_m = np.linalg.norm(orbit.state[:, :3], axis=1)
    speeds_m_s = np.linalg.norm(orbit.state[:, 3:6], axis=1)
    expected_m_s = np.sqrt(EARTH_MU_M3_S2*(2./radii_m-1./semi_major_m))
    energy_m2_s2 = .5*speeds_m_s**2-EARTH_MU_M3_S2/radii_m
    expected_energy_m2_s2 = -EARTH_MU_M3_S2/(2.*semi_major_m)
    worst_energy = int(np.argmax(abs(energy_m2_s2/expected_energy_m2_s2-1.)))
    worst_speed = int(np.argmax(abs(speeds_m_s/expected_m_s-1.)))
    rows.extend([
        BenchmarkResult("Orbital specific energy (worst)", "analytical", float(energy_m2_s2[worst_energy]),
            expected_energy_m2_s2, "m2/s2", 2e-9, "Kepler energy: -mu/(2a)", "One eccentric orbit; no J2, air or thrust."),
        BenchmarkResult("Vis-viva speed (worst)", "analytical", float(speeds_m_s[worst_speed]),
            float(expected_m_s[worst_speed]), "m/s", 2e-9, "v=sqrt(mu*(2/r-1/a))", "Independent specified a; worst relative sample.")])
    for height_m, kind, pressure_pa in ((0., "geopotential", 101325.), (11000., "geopotential", 22632.06),
                                      (20000., "geopotential", 5474.889), (47000., "geopotential", 110.9063),
                                      (86000., "geometric", .37338)):
        rows.append(BenchmarkResult(f"US76 pressure {height_m/1000:g} km", "analytical",
            float(atmosphere(height_m, altitude_kind=kind).pressure_pa), pressure_pa, "Pa", 2e-5,
            US76_SOURCE, f"{kind} height; reference table rounding included in tolerance."))
    return rows


def openrocket_case() -> tuple[VehicleConfig, SimulationConfig, dict[str, object]]:
    """Map the archived OR24.12 two-stage ascent to a declared lumped 3DoF case.

    Returns
    -------
    vehicle, simulation, reference : tuple
        SI inputs and reference outputs/provenance. Reference altitude is AGL.

    Notes
    -----
    Starts at the first stored liftoff event (0.067 s), using recorded position
    and speed. Both stages use recorded thrust, equivalent constant Isp from
    integrated impulse / recorded mass loss, and median pre-apogee stage Cd.
    These shared inputs make this a reduced-order integration comparison;
    aerodynamics are not independently validated. Wind/attitude, variable Cd,
    pad hold and parachutes are not reproduced. Never tune to apogee output.
    """
    path = ROOT/"assets/reference/openrocket_two_stage_24_12.ork"
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("rocket.ork"))
    simulation = root.find("simulations/simulation")
    branch = simulation.find("flightdata/databranch")
    names = branch.attrib["types"].split(",")
    data = np.array([[float(value) for value in row.text.split(",")]
                     for row in branch.findall("datapoint")])
    times_s, mass_kg = data[:, names.index("Time")], data[:, names.index("Mass")]
    events = branch.findall("event")
    start_s = next(float(e.attrib["time"]) for e in events if e.attrib["type"] == "liftoff")
    burns_s = [float(e.attrib["time"]) for e in events if e.attrib["type"] == "burnout"]
    apogee_s = next(float(e.attrib["time"]) for e in events if e.attrib["type"] == "apogee")
    start_index, sep_index, end_index = [int(np.searchsorted(times_s, t)) for t in (start_s, *burns_s)]
    # Use the same sustainer branch for both sides; avoid mixing separated branches.
    upper_fuel_kg = mass_kg[sep_index]-mass_kg[end_index]
    booster_fuel_kg = mass_kg[start_index]-mass_kg[sep_index-1]
    booster_dry_kg = mass_kg[sep_index-1]-mass_kg[sep_index]
    stages: list[StageConfig] = []
    for index, (begin, end, fuel_kg, dry_kg) in enumerate((
            (start_index, sep_index, booster_fuel_kg, booster_dry_kg),
            (sep_index, end_index, upper_fuel_kg, mass_kg[end_index]))):
        local_time_s = times_s[begin:end+1]-times_s[begin]
        thrust_n = data[begin:end+1, names.index("Thrust")].copy()
        thrust_n[-1] = 0.
        impulse_n_s = float(np.trapezoid(thrust_n, local_time_s))
        cd_mask = (times_s >= times_s[begin]) & (times_s < (burns_s[0] if index == 0 else apogee_s))
        cd = float(np.median(data[cd_mask, names.index("Drag coefficient")]))
        stages.append(StageConfig(f"OR equivalent stage {index+1}", float(dry_kg), float(fuel_kg),
            impulse_n_s/(fuel_kg*G0_M_S2), 0., thrust_curve_time_s=tuple(local_time_s),
            thrust_curve_vac_n=tuple(thrust_n), reference_area_m2=float(data[begin, names.index("Reference area")]),
            aero=AeroConfig(base_cd=cd, wave_cd=0., hypersonic_cd=cd, lift_slope_per_rad=0.)))
    config = SimulationConfig(duration_s=15.-start_s, max_step_s=.05, rtol=1e-9,
        latitude_rad=float(np.deg2rad(32.367)), longitude_rad=float(np.deg2rad(-106.83)),
        launch_altitude_m=float(data[start_index, names.index("Altitude")]),
        initial_velocity_enu_m_s=(0., 0., float(data[start_index, names.index("Vertical velocity")])),
        rotating_earth=False, j2_enabled=False)
    return VehicleConfig("OpenRocket reduced-order two-stage", tuple(stages), 0.), config, {
        "source": OPENROCKET_SOURCE, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "creator": root.attrib["creator"], "simulation": simulation.findtext("name"),
        "apogee_m": float(simulation.find("flightdata").attrib["maxaltitude"]),
        "max_speed_m_s": float(simulation.find("flightdata").attrib["maxvelocity"]),
        "time_to_apogee_s": apogee_s-start_s, "epoch_offset_s": start_s,
        "time_s": times_s[times_s <= apogee_s]-start_s,
        "altitude_m": data[times_s <= apogee_s, names.index("Altitude")]}


def openrocket_benchmarks() -> tuple[list[BenchmarkResult], SimulationResult, dict[str, object]]:
    """Compare the reduced-order ascent with archived OR outputs at a strict 4% target."""
    vehicle, config, reference = openrocket_case()
    result = simulate(vehicle, config)
    metrics = trajectory_metrics(result, np.pi/2.)
    notes = ("Shared recorded thrust/mass and median stage Cd; vertical/no wind, no parachutes. "
             "Rounded mass immediately before separation; not a full geometry or flight validation.")
    rows = [BenchmarkResult("OpenRocket two-stage apogee", "cross-tool", metrics.apogee_altitude_m,
            float(reference["apogee_m"]), "m", .04, OPENROCKET_SOURCE, notes),
            BenchmarkResult("OpenRocket maximum speed", "cross-tool", float(np.max(result.telemetry["relative_speed_m_s"])),
            float(reference["max_speed_m_s"]), "m/s", .04, OPENROCKET_SOURCE, notes)]
    return rows, result, reference


def public_milestones() -> list[BenchmarkResult]:
    """Compare class-preset solver events to NASA's approximate public schedules.

    Notes
    -----
    Inputs are deliberately unchanged from the repository presets. Launch site,
    payload, throttle, atmosphere and orbital target are unmatched; even a small
    difference is CONTEXT ONLY. The 4% target is shown, not falsely certified.
    """
    rows: list[BenchmarkResult] = []
    for key, source, references in (
        (list(PRESETS)[1], ELECTRON_SOURCE, {"MECO": 149., "stage_1_separation": 153., "SES-1": 156., "SECO": 567.}),
        (list(PRESETS)[2], FALCON_SOURCE, {"MECO": 156., "stage_1_separation": 159., "SES-1": 167., "SECO": 527.})):
        result = simulate(PRESETS[key], SimulationConfig(duration_s=600., max_step_s=5.),
                          guidance=GuidanceProgram(SteeringConfig(target=OrbitTarget())))
        recorded = {event.name: event.time_s for event in result.events}
        label = "Electron / TROPICS" if "Electron" in key else "Falcon 9 / Crew-2"
        for name, reference_s in references.items():
            rows.append(BenchmarkResult(f"{label} {name}", "contextual", recorded.get(name, np.nan),
                reference_s, "s", .04, source,
                "Approximate published schedule, not measured telemetry. Unmatched payload, throttle, site and target; no observational validation claim."))
    return rows


def run_validation() -> list[BenchmarkResult]:
    """Return freshly executed analytical, cross-tool and contextual benchmark rows."""
    return analytical_benchmarks()+openrocket_benchmarks()[0]+public_milestones()
