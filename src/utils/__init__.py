"""Utility functions for spatial math, geodesic metrics, search theory evaluation, and paleoclimate corrections."""

from src.utils.geo import (
    KNOTS_TO_M_S,
    M_S_TO_KNOTS,
    METERS_PER_DEGREE_LAT,
    haversine_bearing_deg,
    haversine_distance_m,
    knots_to_m_s,
    lat_lon_to_meters_offsets,
    m_s_to_knots,
    meters_to_lat_lon_offsets,
)
from src.utils.metrics import (
    BenchmarkMetrics,
    compute_centroid_error,
    compute_containment_status,
    compute_search_area_reduction,
)

__all__ = [
    # Geodesic
    "KNOTS_TO_M_S",
    "M_S_TO_KNOTS",
    "METERS_PER_DEGREE_LAT",
    "haversine_bearing_deg",
    "haversine_distance_m",
    "knots_to_m_s",
    "lat_lon_to_meters_offsets",
    "m_s_to_knots",
    "meters_to_lat_lon_offsets",
    # Search metrics
    "BenchmarkMetrics",
    "compute_centroid_error",
    "compute_containment_status",
    "compute_search_area_reduction",
]
