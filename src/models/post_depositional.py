"""Post-depositional site formation and multi-century storm dispersal model (Muckelroy, 1978)."""

from dataclasses import dataclass, field
import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.models.debris_field import DebrisField, generate_confidence_ellipse_points
from src.modules.extreme_event import TropicalCyclone
from src.utils.geo import (
    METERS_PER_DEGREE_LAT,
    haversine_bearing_deg,
    haversine_distance_m,
    lat_lon_to_meters_offsets,
    meters_to_lat_lon_offsets,
)

# Seawater and sediment constants
RHO_SEAWATER = 1025.0
GRAVITY = 9.80665


def solve_wave_wavenumber(wave_period_s: float, water_depth_m: float) -> float:
    """Solve the linear wave dispersion relation omega^2 = g * k * tanh(k * d) for wavenumber k.

    Uses Newton-Raphson iteration converging to machine precision.

    Args:
        wave_period_s: Peak wave period T_p in seconds.
        water_depth_m: Water depth in meters (must be > 0).

    Returns:
        Wavenumber k in radians per meter (m^-1).
    """
    if water_depth_m <= 0.0:
        raise ValueError(f"water_depth_m must be strictly positive, got {water_depth_m}")
    if wave_period_s <= 0.0:
        raise ValueError(f"wave_period_s must be strictly positive, got {wave_period_s}")

    omega = 2.0 * math.pi / wave_period_s
    k0 = (omega**2) / GRAVITY  # Deep water initial guess

    k = k0 / math.sqrt(math.tanh(k0 * water_depth_m))  # Initial approximation

    for _ in range(10):
        th = math.tanh(k * water_depth_m)
        f_val = (omega**2) - GRAVITY * k * th
        f_prime = -GRAVITY * (th + k * water_depth_m * (1.0 - th**2))
        step = f_val / f_prime
        k -= step
        if abs(step) < 1e-8:
            break

    return max(1e-6, k)


def compute_bottom_orbital_velocity(
    wave_height_hs_m: float,
    wave_period_tp_s: float,
    water_depth_m: float,
) -> float:
    """Compute near-bed wave orbital velocity amplitude u_b using linear Airy wave theory.

    Formula:
        u_b = (pi * H_s) / (T_p * sinh(k * d))

    Args:
        wave_height_hs_m: Significant wave height in meters.
        wave_period_tp_s: Peak wave period in seconds.
        water_depth_m: Water depth in meters.

    Returns:
        Bottom orbital velocity u_b in meters per second.
    """
    if wave_height_hs_m <= 0.0 or water_depth_m <= 0.0:
        return 0.0

    k = solve_wave_wavenumber(wave_period_tp_s, water_depth_m)
    kd = k * water_depth_m

    # In deep water (kd > 15), sinh(kd) explodes; orbital motion at bottom is zero
    if kd > 15.0:
        return 0.0

    sinh_kd = math.sinh(kd)
    if sinh_kd < 1e-6:
        return 0.0

    ub = (math.pi * wave_height_hs_m) / (wave_period_tp_s * sinh_kd)
    return ub


@dataclass
class StormImpactEvent:
    """Quantifies seafloor hydrodynamic disturbance from a single tropical cyclone passage."""

    storm_name: str
    year: int
    closest_approach_km: float
    wind_speed_ms: float
    wave_height_hs_m: float
    bottom_orbital_velocity_ms: float
    bed_shear_stress_pa: float
    light_artifacts_mobilized: bool
    surge_bearing_deg: float
    scatter_displacement_m: float
    ballast_scour_depth_m: float


@dataclass
class PostDepositionalSiteState:
    """Multi-century post-depositional state of a submerged wreck site."""

    # 1. Initial Touchdown Point (Primary Ballast Mound)
    touchdown_lat: float
    touchdown_lon: float
    water_depth_m: float
    year_of_loss: int

    # 2. Cumulative Ballast Mound Scour & Burial
    ballast_scour_depth_m: float = 0.0

    # 3. Mobile Secondary Artifact Scatter Trail
    scatter_centroid_lat: float = 0.0
    scatter_centroid_lon: float = 0.0
    scatter_semi_major_m: float = 30.0
    scatter_semi_minor_m: float = 15.0
    scatter_orientation_deg: float = 0.0
    cumulative_net_displacement_m: float = 0.0

    # 4. Chronological Storm Impact History
    storm_events: List[StormImpactEvent] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        """Generate statistical summary of post-depositional site transformation."""
        mobilized_storms = [s for s in self.storm_events if s.light_artifacts_mobilized]
        return {
            "initial_touchdown_coords": {
                "lat": round(self.touchdown_lat, 5),
                "lon": round(self.touchdown_lon, 5),
            },
            "current_scatter_centroid": {
                "lat": round(self.scatter_centroid_lat, 5),
                "lon": round(self.scatter_centroid_lon, 5),
            },
            "net_artifact_migration_m": round(self.cumulative_net_displacement_m, 1),
            "ballast_mound_scour_depth_m": round(self.ballast_scour_depth_m, 2),
            "scatter_ellipse_dimensions_m": {
                "semi_major": round(self.scatter_semi_major_m, 1),
                "semi_minor": round(self.scatter_semi_minor_m, 1),
                "orientation_deg": round(self.scatter_orientation_deg, 1),
            },
            "total_storms_analyzed": len(self.storm_events),
            "mobilizing_storm_events": len(mobilized_storms),
        }

    def to_geojson(self) -> Dict[str, Any]:
        """Export initial touchdown and multi-century secondary scatter as distinct GeoJSON features."""
        features: List[Dict[str, Any]] = []

        # Feature 1: Initial Touchdown Point (Primary Ballast Mound Point)
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [round(self.touchdown_lon, 6), round(self.touchdown_lat, 6)],
            },
            "properties": {
                "feature_type": "initial_touchdown_site",
                "site_component": "primary_ballast_mound_core",
                "year_of_loss": self.year_of_loss,
                "water_depth_m": self.water_depth_m,
                "scour_depression_m": round(self.ballast_scour_depth_m, 2),
                "status": "immobile_scoured_in_place",
            },
        })

        # Helper to convert local meter ellipse to [lon, lat] polygon points
        def _build_polygon(origin_lat: float, origin_lon: float, major: float, minor: float, orient: float) -> List[List[float]]:
            meter_pts = generate_confidence_ellipse_points(
                center_x_m=0.0,
                center_y_m=0.0,
                semi_major_m=major,
                semi_minor_m=minor,
                orientation_deg=orient,
                confidence_level=0.95,
            )
            pts = []
            for dx, dy in meter_pts:
                d_lat, d_lon = meters_to_lat_lon_offsets(dx, dy, origin_lat)
                pts.append([round(origin_lon + d_lon, 6), round(origin_lat + d_lat, 6)])
            return pts

        # Feature 2: Compact Primary Ballast Mound Confidence Contour (e.g. 50m x 20m)
        ballast_polygon_coords = _build_polygon(
            origin_lat=self.touchdown_lat,
            origin_lon=self.touchdown_lon,
            major=25.0,
            minor=12.0,
            orient=self.scatter_orientation_deg,
        )
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [ballast_polygon_coords],
            },
            "properties": {
                "feature_type": "ballast_mound_95_percent_contour",
                "site_component": "primary_iron_and_stone_core",
                "status": "anchored_touchdown_location",
            },
        })

        # Feature 3: Present-Day Migrated Secondary Artifact Scatter Centroid
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [round(self.scatter_centroid_lon, 6), round(self.scatter_centroid_lat, 6)],
            },
            "properties": {
                "feature_type": "migrated_artifact_scatter_centroid",
                "site_component": "secondary_coin_and_cargo_dispersion",
                "net_displacement_m": round(self.cumulative_net_displacement_m, 1),
                "status": "post_depositional_storm_migration",
            },
        })

        # Feature 4: Present-Day Migrated Secondary Artifact Scatter 95% Contour Polygon
        scatter_polygon_coords = _build_polygon(
            origin_lat=self.scatter_centroid_lat,
            origin_lon=self.scatter_centroid_lon,
            major=self.scatter_semi_major_m,
            minor=self.scatter_semi_minor_m,
            orient=self.scatter_orientation_deg,
        )
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [scatter_polygon_coords],
            },
            "properties": {
                "feature_type": "secondary_scatter_95_percent_contour",
                "site_component": "wide_magnetometer_and_metal_detection_corridor",
                "semi_major_m": round(self.scatter_semi_major_m, 1),
                "semi_minor_m": round(self.scatter_semi_minor_m, 1),
            },
        })

        # Feature 5: Net Migration LineString connecting Primary Mound to Scatter Centroid
        if self.cumulative_net_displacement_m > 1.0:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [round(self.touchdown_lon, 6), round(self.touchdown_lat, 6)],
                        [round(self.scatter_centroid_lon, 6), round(self.scatter_centroid_lat, 6)],
                    ],
                },
                "properties": {
                    "feature_type": "storm_driven_migration_vector",
                    "displacement_distance_m": round(self.cumulative_net_displacement_m, 1),
                },
            })

        return {"type": "FeatureCollection", "features": features}

    def to_geojson_str(self, indent: int = 2) -> str:
        """Export GeoJSON as formatted string."""
        return json.dumps(self.to_geojson(), indent=indent)


class PostDepositionalDispersalModel:
    """Simulates secondary seafloor artifact dispersal and ballast scour across multi-century storm chronologies."""

    def __init__(
        self,
        artifact_drag_factor: float = 0.015,
        coin_mobilization_threshold_ms: float = 0.80,
    ) -> None:
        """Initialize post-depositional dispersal simulator.

        Args:
            artifact_drag_factor: Efficiency factor converting near-bed wave orbital energy
                into horizontal bedload displacement (typically 0.01 - 0.03).
            coin_mobilization_threshold_ms: Critical bottom orbital velocity u_b (m/s)
                required to initiate saltation and rolling of silver/gold coins (~0.8 m/s).
        """
        self.artifact_drag_factor = artifact_drag_factor
        self.coin_mobilization_threshold_ms = coin_mobilization_threshold_ms

    def simulate_storm_chronology(
        self,
        initial_touchdown_lat: float,
        initial_touchdown_lon: float,
        water_depth_m: float,
        year_of_loss: int,
        historical_storms: List[TropicalCyclone],
    ) -> PostDepositionalSiteState:
        """Replay historical tropical cyclones from loss year to present, simulating seafloor dispersal.

        Args:
            initial_touchdown_lat: Seafloor resting latitude from forward drift / backtracker.
            initial_touchdown_lon: Seafloor resting longitude.
            water_depth_m: Water depth at wreck site in meters.
            year_of_loss: Year in which the vessel foundered (e.g. 1622).
            historical_storms: Chronological list of TropicalCyclone instances passing through region.

        Returns:
            PostDepositionalSiteState capturing initial touchdown and present-day migrated scatter.
        """
        site = PostDepositionalSiteState(
            touchdown_lat=initial_touchdown_lat,
            touchdown_lon=initial_touchdown_lon,
            water_depth_m=water_depth_m,
            year_of_loss=year_of_loss,
            scatter_centroid_lat=initial_touchdown_lat,
            scatter_centroid_lon=initial_touchdown_lon,
            scatter_semi_major_m=40.0,
            scatter_semi_minor_m=20.0,
            scatter_orientation_deg=0.0,
        )

        cur_dx_m = 0.0
        cur_dy_m = 0.0

        for cyclone in historical_storms:
            if cyclone.year < year_of_loss:
                continue

            dist_km = haversine_distance_m(
                site.touchdown_lat, site.touchdown_lon, cyclone.eye_lat, cyclone.eye_lon
            ) / 1000.0

            # Only storms within 150 km exert meaningful hydrodynamic wave influence on the bottom
            if dist_km > 150.0:
                continue

            # 1. Compute surface wind and parametric wave field at site
            u_w, v_w, _ = cyclone.evaluate_at_point(site.touchdown_lat, site.touchdown_lon)
            wind_speed = math.hypot(u_w, v_w)

            # Significant wave height from wind duration/fetch scaling
            hs = min(14.0, 0.24 * (wind_speed**1.2))
            tp = max(6.0, 4.0 * math.sqrt(hs))

            # 2. Bottom orbital velocity at water_depth_m
            ub = compute_bottom_orbital_velocity(
                wave_height_hs_m=hs,
                wave_period_tp_s=tp,
                water_depth_m=water_depth_m,
            )

            # Bed friction factor and shear stress tau = 0.5 * rho * f_w * ub^2
            f_w = 0.025
            bed_shear_pa = 0.5 * RHO_SEAWATER * f_w * (ub**2)

            # Surge heading aligns with wind propagation direction
            surge_heading_deg = (math.degrees(math.atan2(u_w, v_w)) + 360.0) % 360.0

            # 3. Check mobilization threshold
            is_mobilized = ub >= self.coin_mobilization_threshold_ms

            step_disp_m = 0.0
            step_scour_m = 0.0

            if is_mobilized:
                # Effective duration of peak storm surge (approx. 4 to 8 hours)
                surge_duration_s = 6.0 * 3600.0
                step_disp_m = ub * surge_duration_s * self.artifact_drag_factor

                rad = math.radians(surge_heading_deg)
                cur_dx_m += step_disp_m * math.sin(rad)
                cur_dy_m += step_disp_m * math.cos(rad)

                # Expand scatter dispersal ellipse
                site.scatter_semi_major_m += step_disp_m * 0.25
                site.scatter_semi_minor_m += step_disp_m * 0.10
                site.scatter_orientation_deg = surge_heading_deg

                # Heavy ballast mound scour depression
                step_scour_m = min(0.3, 0.02 * (ub**1.5))
                site.ballast_scour_depth_m += step_scour_m

            site.storm_events.append(
                StormImpactEvent(
                    storm_name=cyclone.storm_name,
                    year=cyclone.year,
                    closest_approach_km=round(dist_km, 1),
                    wind_speed_ms=round(wind_speed, 1),
                    wave_height_hs_m=round(hs, 1),
                    bottom_orbital_velocity_ms=round(ub, 2),
                    bed_shear_stress_pa=round(bed_shear_pa, 1),
                    light_artifacts_mobilized=is_mobilized,
                    surge_bearing_deg=round(surge_heading_deg, 1),
                    scatter_displacement_m=round(step_disp_m, 1),
                    ballast_scour_depth_m=round(step_scour_m, 3),
                )
            )

        # Update final scatter centroid coordinates
        net_dist = math.hypot(cur_dx_m, cur_dy_m)
        site.cumulative_net_displacement_m = net_dist

        dlat, dlon = meters_to_lat_lon_offsets(cur_dx_m, cur_dy_m, site.touchdown_lat)
        site.scatter_centroid_lat = site.touchdown_lat + dlat
        site.scatter_centroid_lon = site.touchdown_lon + dlon

        return site
