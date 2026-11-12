"""SI constants for the COESA atmosphere and WGS84 Earth.

Notes
-----
Flattening, J2, heat-capacity ratio, and quaternion components are dimensionless.
The atmosphere's reference radius is deliberately distinct from WGS84's radius.
Sources and model limits are recorded in ``docs/physics_models.md``.
"""

G0_M_S2: float = 9.80665
WGS84_A_M: float = 6_378_137.0
WGS84_F: float = 1.0 / 298.257223563
WGS84_B_M: float = WGS84_A_M * (1.0 - WGS84_F)
EARTH_MU_M3_S2: float = 3.986004418e14
EARTH_J2: float = 1.08262998905e-3
EARTH_OMEGA_RAD_S: float = 7.292115e-5
ATMOSPHERE_RADIUS_M: float = 6_356_766.0
AIR_R_J_KG_K: float = 8314.32 / 28.9644
AIR_GAMMA: float = 1.4
