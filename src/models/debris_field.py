"""Bivariate Gaussian debris field scatter and archaeological site footprint generator."""

from dataclasses import dataclass, field
import json
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.profiles.vessel_class import VesselProfile

# Meters per degree latitude (approximate on WGS84 sphere)
METERS_PER_DEG_LAT = 111139.0


def meters_to_lat_lon_offsets(
    dx_meters: float, dy_meters: float, origin_lat: float
) -> Tuple[float, float]:
    """Convert local Cartesian meter offsets (East, North) to (delta_lat, delta_lon) in degrees.

    Args:
        dx_meters: Eastward offset in meters.
        dy_meters: Northward offset in meters.
        origin_lat: Reference latitude in degrees.

    Returns:
        Tuple of (delta_lat_deg, delta_lon_deg).
    """
    d_lat = dy_meters / METERS_PER_DEG_LAT
    cos_lat = math.cos(math.radians(origin_lat))
    # Prevent division by zero near poles
    cos_lat = max(0.01, abs(cos_lat))
    d_lon = dx_meters / (METERS_PER_DEG_LAT * cos_lat)
    return d_lat, d_lon


def generate_confidence_ellipse_points(
    center_x_m: float,
    center_y_m: float,
    semi_major_m: float,
    semi_minor_m: float,
    orientation_deg: float,
    confidence_level: float = 0.95,
    num_points: int = 36,
) -> List[Tuple[float, float]]:
    """Compute (x, y) coordinates for a bivariate Gaussian confidence ellipse in local meters.

    The radius multiplier for confidence p in 2D is:
        r = sqrt(-2 * ln(1 - p))

    Args:
        center_x_m: Center Eastward coordinate in meters.
        center_y_m: Center Northward coordinate in meters.
        semi_major_m: 1-sigma semi-major axis in meters.
        semi_minor_m: 1-sigma semi-minor axis in meters.
        orientation_deg: Compass bearing of major axis (degrees clockwise from North).
        confidence_level: Probability mass enclosed (e.g. 0.50, 0.80, 0.95).
        num_points: Number of polygon boundary vertices.

    Returns:
        List of (x_m, y_m) polygon vertices closing on the first point.
    """
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")

    scale = math.sqrt(-2.0 * math.log(1.0 - confidence_level))
    a = semi_major_m * scale
    b = semi_minor_m * scale

    # Compass heading: 0 = North (+y), 90 = East (+x)
    # Standard math theta from positive x-axis (East): theta = 90 - orientation
    heading_rad = math.radians(orientation_deg % 360.0)

    points: List[Tuple[float, float]] = []
    for i in range(num_points):
        phi = 2.0 * math.pi * float(i) / float(num_points)
        # Unrotated ellipse coordinates (x' along major axis, y' along minor axis)
        x_prime = a * math.cos(phi)
        y_prime = b * math.sin(phi)

        # Rotate clockwise by heading_rad relative to North:
        # x = center_x + x' * cos(heading) + y' * sin(heading)
        # y = center_y + x' * -sin(heading) + y' * cos(heading) ... using 2D rotation:
        # Major axis aligns with heading: (sin(heading), cos(heading))
        # Minor axis aligns perpendicular: (cos(heading), -sin(heading))
        x = center_x_m + (x_prime * math.sin(heading_rad) + y_prime * math.cos(heading_rad))
        y = center_y_m + (x_prime * math.cos(heading_rad) - y_prime * math.sin(heading_rad))
        points.append((x, y))

    # Close the polygon loop
    points.append(points[0])
    return points


@dataclass
class DebrisField:
    """Archaeological site formation model representing a two-component shipwreck debris field.

    Attributes:
        touchdown_lat: Seafloor latitude of primary impact / ballast mound.
        touchdown_lon: Seafloor longitude of primary impact / ballast mound.
        ballast_semi_major_m: 1-sigma major radius of dense ballast mound in meters.
        ballast_semi_minor_m: 1-sigma minor radius of dense ballast mound in meters.
        ballast_heading_deg: Heading orientation of ship's keel on impact.
        scatter_center_lat: Latitude of secondary lighter debris scatter centroid.
        scatter_center_lon: Longitude of secondary lighter debris scatter centroid.
        scatter_semi_major_m: 1-sigma major radius of downstream debris scatter in meters.
        scatter_semi_minor_m: 1-sigma minor radius of downstream debris scatter in meters.
        current_heading_deg: Heading orientation of deep ocean current driving debris dispersal.
        confidence_levels: List of probability contours calculated [0.50, 0.80, 0.95].
        metadata: Extra archaeological properties (vessel name, depth, descent time).
    """

    touchdown_lat: float
    touchdown_lon: float
    ballast_semi_major_m: float
    ballast_semi_minor_m: float
    ballast_heading_deg: float
    scatter_center_lat: float
    scatter_center_lon: float
    scatter_semi_major_m: float
    scatter_semi_minor_m: float
    current_heading_deg: float
    confidence_levels: List[float] = field(default_factory=lambda: [0.50, 0.80, 0.95])
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_contour_coordinates(
        self, component: str, confidence_level: float = 0.95
    ) -> List[List[float]]:
        """Compute polygon coordinates in [lon, lat] format for specified component and confidence.

        Args:
            component: 'ballast' for primary dense mound, 'scatter' for light artifact trail.
            confidence_level: Probability mass enclosed (default 0.95).

        Returns:
            List of [longitude, latitude] coordinate pairs.
        """
        if component == "ballast":
            origin_lat, origin_lon = self.touchdown_lat, self.touchdown_lon
            major, minor = self.ballast_semi_major_m, self.ballast_semi_minor_m
            heading = self.ballast_heading_deg
        elif component == "scatter":
            origin_lat, origin_lon = self.scatter_center_lat, self.scatter_center_lon
            major, minor = self.scatter_semi_major_m, self.scatter_semi_minor_m
            heading = self.current_heading_deg
        else:
            raise ValueError(f"Unknown component '{component}'. Valid: 'ballast', 'scatter'.")

        meter_pts = generate_confidence_ellipse_points(
            center_x_m=0.0,
            center_y_m=0.0,
            semi_major_m=major,
            semi_minor_m=minor,
            orientation_deg=heading,
            confidence_level=confidence_level,
        )

        geo_coords: List[List[float]] = []
        for dx, dy in meter_pts:
            d_lat, d_lon = meters_to_lat_lon_offsets(dx, dy, origin_lat)
            geo_coords.append([round(origin_lon + d_lon, 6), round(origin_lat + d_lat, 6)])

        return geo_coords

    def area_sq_meters(self, component: str = "ballast", confidence_level: float = 0.95) -> float:
        """Calculate enclosed geometric area of confidence ellipse in square meters."""
        scale = math.sqrt(-2.0 * math.log(1.0 - confidence_level))
        if component == "ballast":
            return math.pi * (self.ballast_semi_major_m * scale) * (self.ballast_semi_minor_m * scale)
        elif component == "scatter":
            return math.pi * (self.scatter_semi_major_m * scale) * (self.scatter_semi_minor_m * scale)
        else:
            raise ValueError(f"Unknown component '{component}'")

    def to_geojson(self) -> Dict[str, Any]:
        """Export debris field features into a GeoJSON FeatureCollection."""
        features: List[Dict[str, Any]] = [
            # 1. Primary Ballast Mound Centroid
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(self.touchdown_lon, 6), round(self.touchdown_lat, 6)],
                },
                "properties": {
                    "feature_type": "primary_ballast_touchdown",
                    "description": "Dense artifact concentration (cannon, anchors, ballast stones)",
                    **self.metadata,
                },
            },
            # 2. Ballast Mound 95% Confidence Ellipse
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [self.get_contour_coordinates("ballast", 0.95)],
                },
                "properties": {
                    "feature_type": "ballast_mound_95_percent_contour",
                    "confidence_level": 0.95,
                    "area_m2": round(self.area_sq_meters("ballast", 0.95), 1),
                },
            },
            # 3. Secondary Debris Scatter 95% Confidence Ellipse
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [self.get_contour_coordinates("scatter", 0.95)],
                },
                "properties": {
                    "feature_type": "debris_scatter_trail_95_percent_contour",
                    "confidence_level": 0.95,
                    "area_m2": round(self.area_sq_meters("scatter", 0.95), 1),
                },
            },
        ]

        return {
            "type": "FeatureCollection",
            "features": features,
        }

    def to_geojson_str(self, indent: int = 2) -> str:
        """Export debris field GeoJSON as formatted string."""
        return json.dumps(self.to_geojson(), indent=indent)


def estimate_debris_field(
    touchdown_lat: float,
    touchdown_lon: float,
    deep_current_u_m_s: float,
    deep_current_v_m_s: float,
    descent_duration_s: float,
    profile: VesselProfile,
    keel_impact_heading_deg: float = 45.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> DebrisField:
    """Generate an archaeological debris field from 3D descent physics and vessel profile.

    Args:
        touchdown_lat: Latitude where hull/ballast first hits seafloor.
        touchdown_lon: Longitude where hull/ballast first hits seafloor.
        deep_current_u_m_s: Eastward deep-water ocean current in m/s.
        deep_current_v_m_s: Northward deep-water ocean current in m/s.
        descent_duration_s: Time in seconds vessel spent descending through water column.
        profile: VesselProfile defining dimensions, ballast ratio, and modern/wooden status.
        keel_impact_heading_deg: Heading orientation of the ship at moment of foundering.
        metadata: Optional dictionary of metadata attributes.

    Returns:
        DebrisField instance.
    """
    # 1. Primary Ballast Mound dimensions (dense artifacts drop compactly)
    # Historical galleons (~40m length, 10m beam); modern bulkers (~200m length, 30m beam)
    if profile.is_modern:
        ballast_major_1sigma = 90.0
        ballast_minor_1sigma = 25.0
    else:
        ballast_major_1sigma = 25.0
        ballast_minor_1sigma = 12.0

    # 2. Downstream Debris Scatter Trail
    # Lighter cargo, broken masts, rigging drift down-current during descent and early post-depositional scour
    curr_speed = math.hypot(deep_current_u_m_s, deep_current_v_m_s)
    if curr_speed > 0.0:
        curr_heading_rad = math.atan2(deep_current_u_m_s, deep_current_v_m_s)
        curr_heading_deg = math.degrees(curr_heading_rad) % 360.0
    else:
        curr_heading_deg = keel_impact_heading_deg

    # Downstream drift displacement for buoyant/shattered debris during descent
    # Lighter debris has lower settling velocity (e.g. 50% of hull descent speed),
    # meaning it stays in current twice as long
    drift_duration = descent_duration_s * 1.5
    drift_dist_m = curr_speed * drift_duration

    dx_drift = drift_dist_m * math.sin(math.radians(curr_heading_deg))
    dy_drift = drift_dist_m * math.cos(math.radians(curr_heading_deg))

    d_lat, d_lon = meters_to_lat_lon_offsets(dx_drift, dy_drift, touchdown_lat)
    scatter_lat = touchdown_lat + d_lat
    scatter_lon = touchdown_lon + d_lon

    # Debris scatter elongation scales with current speed and water depth / duration
    scatter_major_1sigma = max(ballast_major_1sigma * 2.0, drift_dist_m * 0.4)
    scatter_minor_1sigma = max(ballast_minor_1sigma * 1.5, drift_dist_m * 0.15)

    meta = metadata or {}
    meta.update(
        {
            "vessel_class": profile.class_name,
            "is_modern": profile.is_modern,
            "descent_duration_s": round(descent_duration_s, 1),
            "deep_current_speed_m_s": round(curr_speed, 3),
        }
    )

    return DebrisField(
        touchdown_lat=touchdown_lat,
        touchdown_lon=touchdown_lon,
        ballast_semi_major_m=ballast_major_1sigma,
        ballast_semi_minor_m=ballast_minor_1sigma,
        ballast_heading_deg=keel_impact_heading_deg,
        scatter_center_lat=scatter_lat,
        scatter_center_lon=scatter_lon,
        scatter_semi_major_m=scatter_major_1sigma,
        scatter_semi_minor_m=scatter_minor_1sigma,
        current_heading_deg=curr_heading_deg,
        metadata=meta,
    )
