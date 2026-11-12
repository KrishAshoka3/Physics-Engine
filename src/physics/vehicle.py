"""Typed vehicle inputs and openly documented class-level example configurations.

Notes
-----
Presets are engineering estimates, not manufacturer digital twins or guaranteed
performance. Mass breakdowns, inertias, exit areas, and drag need mission-specific
calibration. References and assumed quantities are in ``docs/physics_models.md``.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .aerodynamics import AeroConfig
from .constants import G0_M_S2


@dataclass(frozen=True)
class StageConfig:
    """One stage and its attached-stack aerodynamic/rigid-body approximation.

    Parameters
    ----------
    name : str
        Human-readable label (not a physical quantity).
    dry_mass_kg, propellant_mass_kg : float
        Stage-only masses [kg]; upper stages, payload, and fairing are separate.
    isp_vac_s, nozzle_exit_area_m2 : float
        Vacuum specific impulse [s] and total exit area of active nozzles [m^2].
    mass_flow_kg_s : float or None
        Constant positive mass flow [kg/s], exclusive with a thrust curve.
    thrust_curve_time_s, thrust_curve_vac_n : tuple of float
        Piecewise-linear arbitrary vacuum thrust samples [s], [N], starting at
        local ignition time zero. Curve consumption follows F_vac/(Isp*g0).
    cutoff_time_s : float or None
        Scheduled cutoff [s since ignition]; defaults to depletion for constant
        flow or the last curve time. Residual propellant remains until separation.
    separation_delay_s, coast_after_separation_s : float
        Burnout-to-separation and separation-to-next-ignition delays [s].
    reference_area_m2, stack_length_m, stack_radius_m : float
        Aerodynamic area [m^2] and uniform-cylinder attached-stack geometry [m].
    inertia_per_mass_m2 : tuple or None
        Optional principal moments divided by current total mass [m^2]. Body
        +x is the nose axis. Defaults to the uniform-cylinder approximation.
    cp_offset_body_m, thrust_offset_body_m : tuple
        Force application offsets from instantaneous modeled CM [m].
    angular_damping_n_m_s : float
        Linear body-rate damping [N*m*s/rad], default zero.
    aero : AeroConfig
        Dimensionless aerodynamic model parameters.

    Notes
    -----
    Geometry and offsets describe the entire attached stack in this phase, not
    the detached motor. Inertia scales with mass; CM migration and efflux angular
    momentum are not reconstructed from tank geometry.
    """

    name: str
    dry_mass_kg: float
    propellant_mass_kg: float
    isp_vac_s: float
    nozzle_exit_area_m2: float
    mass_flow_kg_s: float | None = None
    thrust_curve_time_s: tuple[float, ...] = ()
    thrust_curve_vac_n: tuple[float, ...] = ()
    cutoff_time_s: float | None = None
    separation_delay_s: float = 0.
    coast_after_separation_s: float = 0.
    reference_area_m2: float = 1.
    stack_length_m: float = 10.
    stack_radius_m: float = .5
    inertia_per_mass_m2: tuple[float, float, float] | None = None
    cp_offset_body_m: tuple[float, float, float] = (0., 0., 0.)
    thrust_offset_body_m: tuple[float, float, float] = (0., 0., 0.)
    angular_damping_n_m_s: float = 0.
    aero: AeroConfig = field(default_factory=AeroConfig)

    def __post_init__(self) -> None:
        """Validate positive SI properties and exclusive propulsion input modes."""
        positive = (self.dry_mass_kg, self.propellant_mass_kg, self.isp_vac_s, self.stack_length_m, self.stack_radius_m)
        nonnegative = (self.nozzle_exit_area_m2, self.separation_delay_s, self.coast_after_separation_s,
                       self.reference_area_m2, self.angular_damping_n_m_s)
        if not np.all(np.isfinite(positive + nonnegative)) or min(positive) <= 0. or min(nonnegative) < 0.:
            raise ValueError("Stage properties must be finite with positive masses, Isp, and dimensions")
        has_curve = bool(self.thrust_curve_time_s or self.thrust_curve_vac_n)
        if has_curve == (self.mass_flow_kg_s is not None):
            raise ValueError("Specify exactly one of constant mass flow or a vacuum thrust curve")
        if has_curve:
            times_s, thrust_n = np.asarray(self.thrust_curve_time_s), np.asarray(self.thrust_curve_vac_n)
            if (times_s.ndim != 1 or times_s.size < 2 or times_s.shape != thrust_n.shape
                    or not np.all(np.isfinite(times_s)) or not np.all(np.isfinite(thrust_n))
                    or times_s[0] != 0. or np.any(np.diff(times_s) <= 0.) or np.any(thrust_n < 0.) or not np.any(thrust_n > 0.)):
                raise ValueError("Thrust curve requires increasing times from zero and nonnegative finite thrust")
            if self.cutoff_time_s is not None and self.cutoff_time_s > times_s[-1]:
                raise ValueError("Scheduled cutoff cannot exceed the thrust curve domain")
        elif not np.isfinite(self.mass_flow_kg_s) or self.mass_flow_kg_s <= 0.:
            raise ValueError("Constant mass flow must be finite and positive")
        if self.cutoff_time_s is not None and (not np.isfinite(self.cutoff_time_s) or self.cutoff_time_s <= 0.):
            raise ValueError("Scheduled cutoff must be finite and positive")
        for offset_m in (self.cp_offset_body_m, self.thrust_offset_body_m):
            if np.shape(offset_m) != (3,) or not np.all(np.isfinite(offset_m)):
                raise ValueError("Force offsets require finite body three-vectors")
        if self.inertia_per_mass_m2 is not None:
            moments_m2 = np.asarray(self.inertia_per_mass_m2)
            if (moments_m2.shape != (3,) or not np.all(np.isfinite(moments_m2)) or np.any(moments_m2 <= 0.)
                    or 2. * np.max(moments_m2) > np.sum(moments_m2)):
                raise ValueError("Principal inertia per mass must be positive and obey triangle inequalities")

    @property
    def nominal_cutoff_s(self) -> float:
        """Return scheduled engine duration [s]; actual fuel depletion can end it earlier."""
        if self.cutoff_time_s is not None:
            return self.cutoff_time_s
        if self.mass_flow_kg_s is not None:
            return self.propellant_mass_kg / self.mass_flow_kg_s
        return self.thrust_curve_time_s[-1]

    @property
    def principal_inertia_per_mass_m2(self) -> tuple[float, float, float]:
        """Return principal inertia divided by total attached mass [m^2]."""
        if self.inertia_per_mass_m2 is not None:
            return self.inertia_per_mass_m2
        axial_m2 = .5 * self.stack_radius_m**2
        transverse_m2 = (3. * self.stack_radius_m**2 + self.stack_length_m**2) / 12.
        return axial_m2, transverse_m2, transverse_m2


@dataclass(frozen=True)
class VehicleConfig:
    """Ordered stages and mission-level payload/fairing inputs in SI units.

    Parameters
    ----------
    name : str
        Vehicle label (not a physical quantity).
    stages : tuple of StageConfig
        Stage order from booster to final stage.
    payload_mass_kg, fairing_mass_kg : float
        Payload and disposable fairing masses [kg].
    fairing_jettison_time_s, fairing_jettison_altitude_m : float or None
        Mutually exclusive absolute elapsed-time [s] or ascending altitude [m]
        trigger. None for both retains the fairing.
    description : str
        Provenance/assumption description (not a physical quantity).
    """

    name: str
    stages: tuple[StageConfig, ...]
    payload_mass_kg: float
    fairing_mass_kg: float = 0.
    fairing_jettison_time_s: float | None = None
    fairing_jettison_altitude_m: float | None = None
    description: str = ""

    def __post_init__(self) -> None:
        """Validate ordered stages, masses [kg], and mutually exclusive fairing triggers."""
        if not self.stages or any(not isinstance(stage, StageConfig) for stage in self.stages):
            raise ValueError("At least one StageConfig is required")
        if not np.all(np.isfinite([self.payload_mass_kg, self.fairing_mass_kg])) or min(self.payload_mass_kg, self.fairing_mass_kg) < 0.:
            raise ValueError("Payload and fairing masses must be finite and nonnegative")
        triggers = (self.fairing_jettison_time_s, self.fairing_jettison_altitude_m)
        if all(trigger is not None for trigger in triggers):
            raise ValueError("Choose time or altitude for fairing jettison")
        if any(trigger is not None and (not np.isfinite(trigger) or trigger < 0.) for trigger in triggers):
            raise ValueError("Fairing triggers must be finite and nonnegative")

    @property
    def initial_mass_kg(self) -> float:
        """Return fully fueled stack, payload, and fairing mass [kg]."""
        return self.payload_mass_kg + self.fairing_mass_kg + sum(stage.dry_mass_kg + stage.propellant_mass_kg for stage in self.stages)


PRESETS: Mapping[str, VehicleConfig] = MappingProxyType({
    "Two-Stage Sounding Rocket (Black Brant IX Class)": VehicleConfig(
        name="Two-Stage Sounding Rocket (Black Brant IX Class)", payload_mass_kg=650., fairing_mass_kg=40.,
        stages=(
            StageConfig("Terrier-class booster", 350., 760., 240., .035, mass_flow_kg_s=760./6.,
                        coast_after_separation_s=6., reference_area_m2=.164, stack_length_m=13., stack_radius_m=.229,
                        thrust_offset_body_m=(-5.2, 0., 0.)),
            StageConfig("Black Brant-class sustainer", 500., 1008., 290., .035, mass_flow_kg_s=1008./27.5,
                        reference_area_m2=.151, stack_length_m=8.7, stack_radius_m=.219, thrust_offset_body_m=(-3.48, 0., 0.))),
        description="Heavy-payload ~200 km sounding class. Upper dry mass includes ~226 kg assumed recovery/structure beyond motor; see model notes."),
    "Small-Lift Orbital Vehicle (Electron Class)": VehicleConfig(
        name="Small-Lift Orbital Vehicle (Electron Class)", payload_mass_kg=200., fairing_mass_kg=44., fairing_jettison_time_s=180.,
        stages=(
            StageConfig("Rutherford-class booster", 950., 9250., 311., .335, mass_flow_kg_s=224000./(311.*G0_M_S2),
                        coast_after_separation_s=3., reference_area_m2=1.131, stack_length_m=18., stack_radius_m=.6,
                        thrust_offset_body_m=(-7.2, 0., 0.)),
            StageConfig("Rutherford-vacuum-class upper stage", 250., 2300., 343., .65, mass_flow_kg_s=25800./(343.*G0_M_S2),
                        reference_area_m2=1.131, stack_length_m=6., stack_radius_m=.6, thrust_offset_body_m=(-2.4, 0., 0.))),
        description="~7.6 km/s LEO mission class; estimated mass allocation; kick stage and battery drops omitted."),
    "Medium-Lift Two-Stage Orbital (Falcon 9 Class)": VehicleConfig(
        name="Medium-Lift Two-Stage Orbital (Falcon 9 Class)", payload_mass_kg=15000., fairing_mass_kg=1900., fairing_jettison_time_s=180.,
        stages=(
            StageConfig("Merlin-class booster", 25600., 395700., 311., 6.91, mass_flow_kg_s=8310000./(311.*G0_M_S2),
                        coast_after_separation_s=3., reference_area_m2=10.752, stack_length_m=70., stack_radius_m=1.85,
                        thrust_offset_body_m=(-28., 0., 0.)),
            StageConfig("Merlin-vacuum-class upper stage", 4000., 92670., 348., 5., mass_flow_kg_s=981000./(348.*G0_M_S2),
                        reference_area_m2=10.752, stack_length_m=16., stack_radius_m=1.85, thrust_offset_body_m=(-6.4, 0., 0.))),
        description="Public thrust/size anchors; estimated expendable mass split and Isp, no recovery reserve or throttle schedule."),
})
