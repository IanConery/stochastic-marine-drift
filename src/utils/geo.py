"""Geospatial utilities, coordinate conversions, and geodesic distance metrics."""

import math
from typing import Tuple

METERS_PER_DEGREE_LAT = 111139.0
KNOTS_TO_M_S = 1852.0 / 3600.0  # 0.514444...
M_S_TO_KNOTS = 3600.0 / 1852.0  # 1.943844...


def knots_to_m_s(knots: float) -> float:
    """Convert speed from knots to meters per second."""
    return knots * KNOTS_TO_M_S


def m_s_to_knots(speed_m_s: float) -> float:
    """Convert speed from meters per second to knots."""
    return speed_m_s * M_S_TO_KNOTS


def meters_to_lat_lon_offsets(
    dx_meters: float, dy_meters: float, reference_lat: float
) -> Tuple[float, float]:
    """Convert horizontal meter offsets (dx East, dy North) to latitude/longitude offsets.

    Args:
        dx_meters: East-West displacement in meters.
        dy_meters: North-South displacement in meters.
        reference_lat: Base latitude in decimal degrees.

    Returns:
        Tuple of (d_lat_deg, d_lon_deg).
    """
    d_lat = dy_meters / METERS_PER_DEGREE_LAT
    cos_lat = math.cos(math.radians(reference_lat))
    if abs(cos_lat) < 1e-6:
        cos_lat = 1e-6
    d_lon = dx_meters / (METERS_PER_DEGREE_LAT * cos_lat)
    return d_lat, d_lon


def lat_lon_to_meters_offsets(
    lat: float, lon: float, reference_lat: float, reference_lon: float
) -> Tuple[float, float]:
    """Convert coordinates (lat, lon) to meter offsets (dx East, dy North) from reference point.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        reference_lat: Reference base latitude in decimal degrees.
        reference_lon: Reference base longitude in decimal degrees.

    Returns:
        Tuple of (dx_meters, dy_meters).
    """
    dy_meters = (lat - reference_lat) * METERS_PER_DEGREE_LAT
    cos_lat = math.cos(math.radians(reference_lat))
    dx_meters = (lon - reference_lon) * METERS_PER_DEGREE_LAT * cos_lat
    return dx_meters, dy_meters


def haversine_distance_m(
    lat1: float, lon1: float, lat2: float, lon2: float, earth_radius_m: float = 6371000.0
) -> float:
    """Compute great-circle distance between two points in meters using the Haversine formula."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return earth_radius_m * c


def haversine_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute forward azimuth bearing from point 1 to point 2 in degrees (0-360 deg)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    initial_bearing = math.atan2(y, x)
    return (math.degrees(initial_bearing) + 360.0) % 360.0
