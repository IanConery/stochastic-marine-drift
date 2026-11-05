"""Search theory verification metrics: Containment Rate (P_contain) and Search Area Reduction (SAR)."""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Tuple
import numpy as np

from src.utils.geo import METERS_PER_DEGREE_LAT, haversine_distance_m, lat_lon_to_meters_offsets


@dataclass
class BenchmarkMetrics:
    """Quantitative performance metrics evaluated against known archaeological ground truth."""

    case_name: str
    target_lat: float
    target_lon: float
    map_centroid_lat: float
    map_centroid_lon: float
    centroid_error_km: float
    centroid_error_nmi: float
    contained_in_50_pct: bool
    contained_in_80_pct: bool
    contained_in_95_pct: bool
    naive_search_area_km2: float
    model_search_area_95_km2: float
    search_area_reduction_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_name": self.case_name,
            "target_coords": {"lat": round(self.target_lat, 5), "lon": round(self.target_lon, 5)},
            "map_centroid": {"lat": round(self.map_centroid_lat, 5), "lon": round(self.map_centroid_lon, 5)},
            "centroid_error_km": round(self.centroid_error_km, 2),
            "centroid_error_nmi": round(self.centroid_error_nmi, 2),
            "containment": {
                "in_50_percent": self.contained_in_50_pct,
                "in_80_percent": self.contained_in_80_pct,
                "in_95_percent": self.contained_in_95_pct,
            },
            "naive_search_area_km2": round(self.naive_search_area_km2, 2),
            "model_search_area_95_km2": round(self.model_search_area_95_km2, 2),
            "search_area_reduction_pct": round(self.search_area_reduction_pct, 2),
        }


def compute_centroid_error(
    predicted_lat: float,
    predicted_lon: float,
    true_lat: float,
    true_lon: float,
) -> Tuple[float, float]:
    """Calculate great-circle error between predicted MAP centroid and ground-truth coordinates.

    Returns:
        Tuple of (error_km, error_nmi).
    """
    error_m = haversine_distance_m(predicted_lat, predicted_lon, true_lat, true_lon)
    error_km = error_m / 1000.0
    error_nmi = error_m / 1852.0
    return error_km, error_nmi


def compute_containment_status(
    simulated_lats: List[float],
    simulated_lons: List[float],
    true_lat: float,
    true_lon: float,
) -> Tuple[bool, bool, bool, float, float]:
    """Evaluate whether true coordinates fall within 50%, 80%, and 95% ensemble spatial contours.

    Args:
        simulated_lats: Array of simulated particle latitudes.
        simulated_lons: Array of simulated particle longitudes.
        true_lat: Ground truth latitude.
        true_lon: Ground truth longitude.

    Returns:
        Tuple of (in_50, in_80, in_95, model_area_95_km2, centroid_lat, centroid_lon).
    """
    if len(simulated_lats) < 3 or len(simulated_lons) < 3:
        raise ValueError("At least 3 particle coordinates required to compute containment.")

    centroid_lat = float(np.mean(simulated_lats))
    centroid_lon = float(np.mean(simulated_lons))

    # Compute Euclidean distance in meters from centroid on local tangent plane
    ref_lat = centroid_lat
    distances_from_center_m = []
    for lat, lon in zip(simulated_lats, simulated_lons):
        dx, dy = lat_lon_to_meters_offsets(lat, lon, ref_lat, centroid_lon)
        distances_from_center_m.append(math.hypot(dx, dy))

    # Ground truth distance from centroid
    true_dx, true_dy = lat_lon_to_meters_offsets(true_lat, true_lon, ref_lat, centroid_lon)
    true_dist_m = math.hypot(true_dx, true_dy)

    # Spatial radius percentiles
    r_50 = float(np.percentile(distances_from_center_m, 50.0))
    r_80 = float(np.percentile(distances_from_center_m, 80.0))
    r_95 = float(np.percentile(distances_from_center_m, 95.0))

    in_50 = bool(true_dist_m <= r_50)
    in_80 = bool(true_dist_m <= r_80)
    in_95 = bool(true_dist_m <= r_95)

    # 95% contour area: pi * r_95^2 in km^2
    area_95_km2 = (math.pi * (r_95**2)) / 1.0e6

    return in_50, in_80, in_95, max(0.1, area_95_km2), centroid_lat, centroid_lon


def compute_search_area_reduction(
    naive_uncertainty_radius_nmi: float,
    model_area_95_km2: float,
) -> Tuple[float, float]:
    """Calculate Search Area Reduction (SAR) percentage relative to naive dead-reckoning circle.

    Args:
        naive_uncertainty_radius_nmi: Radius of initial circular search area in nautical miles.
        model_area_95_km2: Area enclosed by the model's 95% probability contour in km^2.

    Returns:
        Tuple of (naive_area_km2, sar_percentage).
    """
    radius_m = naive_uncertainty_radius_nmi * 1852.0
    naive_area_km2 = (math.pi * (radius_m**2)) / 1.0e6

    if naive_area_km2 <= 0.0:
        return 0.0, 0.0

    sar_pct = max(0.0, min(99.9, (1.0 - (model_area_95_km2 / naive_area_km2)) * 100.0))
    return naive_area_km2, sar_pct
