"""Tropical cyclone forcing via parametric Holland vortex model and IBTrACS integration."""

from dataclasses import dataclass
import math
from typing import Tuple

from src.modules.current import coriolis_parameter, RHO_AIR

# Earth radius in kilometers
EARTH_RADIUS_KM = 6371.0


def haversine_distance_and_bearing(
    lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float
) -> Tuple[float, float]:
    """Calculate great-circle distance (km) and initial compass bearing (deg) from point 1 to point 2.

    Args:
        lat1_deg: Origin latitude in degrees.
        lon1_deg: Origin longitude in degrees.
        lat2_deg: Destination latitude in degrees.
        lon2_deg: Destination longitude in degrees.

    Returns:
        Tuple of (distance_km, bearing_deg_clockwise_from_north).
    """
    phi1 = math.radians(lat1_deg)
    phi2 = math.radians(lat2_deg)
    delta_phi = math.radians(lat2_deg - lat1_deg)
    delta_lambda = math.radians(lon2_deg - lon1_deg)

    # Haversine formula for distance
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2)
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    distance_km = EARTH_RADIUS_KM * c

    # Initial bearing
    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    bearing_rad = math.atan2(y, x)
    bearing_deg = (math.degrees(bearing_rad) + 360.0) % 360.0

    return distance_km, bearing_deg


@dataclass(frozen=True)
class TropicalCyclone:
    """Parametric representation of an intense tropical cyclone (hurricane).

    Attributes:
        storm_name: Identifier for the tropical storm.
        year: Year of storm occurrence.
        eye_lat: Latitude of storm center in degrees.
        eye_lon: Longitude of storm center in degrees.
        v_max_m_s: Maximum sustained 10m wind speed in m/s (1 kt = 0.51444 m/s).
        p_central_hpa: Central minimum barometric pressure in hPa.
        p_ambient_hpa: Ambient environmental pressure outside vortex in hPa.
        r_max_km: Radius of maximum winds (RMW) in kilometers.
        u_trans_m_s: Forward translation speed of storm eye Eastward (m/s).
        v_trans_m_s: Forward translation speed of storm eye Northward (m/s).
        inflow_angle_deg: Boundary layer cross-isobar inflow angle (typically 15° - 25°).
    """

    storm_name: str
    year: int
    eye_lat: float
    eye_lon: float
    v_max_m_s: float
    p_central_hpa: float
    p_ambient_hpa: float = 1013.25
    r_max_km: float = 35.0
    u_trans_m_s: float = 0.0
    v_trans_m_s: float = 0.0
    inflow_angle_deg: float = 20.0

    def holland_b_parameter(self) -> float:
        """Calculate the Holland shape parameter B.

        Formula: B = (rho_air * e * V_max^2) / (100 * delta_P)
        Physically bounded between 1.0 and 2.5 (Holland, 1980).
        """
        delta_p = max(5.0, self.p_ambient_hpa - self.p_central_hpa)
        delta_p_pa = delta_p * 100.0  # Convert hPa to Pa
        b = (RHO_AIR * math.e * (self.v_max_m_s**2)) / delta_p_pa
        return min(max(b, 1.0), 2.5)

    def evaluate_at_point(
        self, target_lat: float, target_lon: float
    ) -> Tuple[float, float, float]:
        """Compute surface wind vector (u, v) in m/s and barometric pressure at a target location.

        Args:
            target_lat: Latitude of target particle in degrees.
            target_lon: Longitude of target particle in degrees.

        Returns:
            Tuple of (u_cyclonic_m_s, v_cyclonic_m_s, pressure_hpa).
        """
        r_km, bearing_eye_to_pt = haversine_distance_and_bearing(
            self.eye_lat, self.eye_lon, target_lat, target_lon
        )

        delta_p = max(5.0, self.p_ambient_hpa - self.p_central_hpa)
        delta_p_pa = delta_p * 100.0
        b = self.holland_b_parameter()

        # Handle singularity in the dead center of the eye
        if r_km < 1.0:
            return self.u_trans_m_s, self.v_trans_m_s, self.p_central_hpa

        # 1. Barometric pressure at radius r: P(r) = P_c + delta_P * exp(-(R_max / r)^B)
        ratio = (self.r_max_km / r_km) ** b
        pressure_hpa = self.p_central_hpa + delta_p * math.exp(-ratio)

        # 2. Gradient cyclonic wind speed
        r_meters = r_km * 1000.0
        f = abs(coriolis_parameter(target_lat))
        # V_grad = sqrt( (B / rho_a) * (R_max/r)^B * delta_P * exp(-(R_max/r)^B) + (r*f/2)^2 ) - (r*f/2)
        term1 = (b / RHO_AIR) * ratio * delta_p_pa * math.exp(-ratio)
        term2 = (r_meters * f / 2.0) ** 2
        v_rotational = math.sqrt(max(0.0, term1 + term2)) - (r_meters * f / 2.0)

        # 3. Wind direction:
        # In Northern Hemisphere, cyclonic flow is counter-clockwise around eye.
        # Tangential heading is (bearing_eye_to_pt - 90 deg).
        # Adding boundary layer inflow towards the eye (-inflow_angle_deg).
        tangential_deg = (bearing_eye_to_pt - 90.0) % 360.0
        inflow_deg = (tangential_deg - self.inflow_angle_deg) % 360.0
        inflow_rad = math.radians(inflow_deg)

        u_rot = v_rotational * math.sin(inflow_rad)
        v_rot = v_rotational * math.cos(inflow_rad)

        # 4. Add forward motion asymmetry (higher winds in right front quadrant in NH)
        # Translation velocity decays far outside storm radius
        trans_weight = math.exp(-0.5 * (r_km / (3.0 * self.r_max_km)))
        u_total = u_rot + self.u_trans_m_s * trans_weight
        v_total = v_rot + self.v_trans_m_s * trans_weight

        return u_total, v_total, pressure_hpa
