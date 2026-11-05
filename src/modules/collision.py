"""Bathymetric collision and reef grounding filter for maritime drift particles."""

from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np


@dataclass
class GroundingEvent:
    """Record of a bathymetric reef collision or grounding event.

    Attributes:
        is_grounded: True if the vessel has impacted the seafloor/reef.
        grounding_lat: Latitude of impact in degrees.
        grounding_lon: Longitude of impact in degrees.
        water_depth_m: Charted water column depth at site in meters.
        effective_depth_m: Depth adjusted for tidal stage and wave trough plunge.
        vessel_draft_m: Keel draft depth of the vessel in meters.
        cause: Physical mechanism ('shallow_shoal', 'reef_collision_wave_trough', 'landfall').
        timestamp_s: Optional simulation timestamp in seconds.
    """

    is_grounded: bool
    grounding_lat: float
    grounding_lon: float
    water_depth_m: float
    effective_depth_m: float
    vessel_draft_m: float
    cause: str
    timestamp_s: Optional[float] = None

    def to_geojson_feature(self) -> Dict[str, Any]:
        """Convert grounding event into a standard GeoJSON Point Feature."""
        return {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [self.grounding_lon, self.grounding_lat],
            },
            "properties": {
                "event_type": "bathymetric_grounding",
                "is_grounded": self.is_grounded,
                "water_depth_m": round(self.water_depth_m, 2),
                "effective_depth_m": round(self.effective_depth_m, 2),
                "vessel_draft_m": round(self.vessel_draft_m, 2),
                "under_keel_clearance_m": round(self.effective_depth_m - self.vessel_draft_m, 2),
                "cause": self.cause,
                "timestamp_s": self.timestamp_s,
            },
        }


@dataclass
class BathymetricGrid:
    """Regular spatial grid of bathymetric depth with bilinear interpolation.

    Attributes:
        lats: 1D array of latitude coordinates in ascending order.
        lons: 1D array of longitude coordinates in ascending order.
        depth_matrix_m: 2D array of shape (len(lats), len(lons)) representing water depth
            in meters (positive = submerged water depth; <= 0 = land/dry reef).
    """

    lats: np.ndarray
    lons: np.ndarray
    depth_matrix_m: np.ndarray

    def __post_init__(self) -> None:
        if self.depth_matrix_m.shape != (len(self.lats), len(self.lons)):
            raise ValueError(
                f"depth_matrix_m shape {self.depth_matrix_m.shape} must match "
                f"({len(self.lats)}, {len(self.lons)})"
            )

    def sample_depth_bilinear(self, lat: float, lon: float) -> float:
        """Sample water depth at (lat, lon) using bilinear interpolation.

        Args:
            lat: Query latitude in degrees.
            lon: Query longitude in degrees.

        Returns:
            Interpolated water depth in meters.
        """
        # Clamp coordinates to grid bounding box
        lat_clamped = min(max(lat, float(self.lats[0])), float(self.lats[-1]))
        lon_clamped = min(max(lon, float(self.lons[0])), float(self.lons[-1]))

        # Find bounding grid indices
        i = int(np.clip(np.searchsorted(self.lats, lat_clamped) - 1, 0, len(self.lats) - 2))
        j = int(np.clip(np.searchsorted(self.lons, lon_clamped) - 1, 0, len(self.lons) - 2))

        lat0, lat1 = self.lats[i], self.lats[i + 1]
        lon0, lon1 = self.lons[j], self.lons[j + 1]

        # Normalized coordinates [0, 1]
        t = (lat_clamped - lat0) / (lat1 - lat0) if (lat1 > lat0) else 0.0
        u = (lon_clamped - lon0) / (lon1 - lon0) if (lon1 > lon0) else 0.0

        # Bilinear interpolation
        q00 = self.depth_matrix_m[i, j]
        q01 = self.depth_matrix_m[i, j + 1]
        q10 = self.depth_matrix_m[i + 1, j]
        q11 = self.depth_matrix_m[i + 1, j + 1]

        depth = (
            (1.0 - t) * (1.0 - u) * q00
            + (1.0 - t) * u * q01
            + t * (1.0 - u) * q10
            + t * u * q11
        )
        return float(depth)


def evaluate_grounding_condition(
    water_depth_m: float,
    vessel_draft_m: float,
    tide_height_m: float = 0.0,
    wave_height_m: float = 0.0,
) -> Tuple[bool, float, str]:
    """Evaluate whether an instantaneous sea state causes keel impact or reef grounding.

    Formula:
        D_eff = water_depth_m + tide_height_m - 0.5 * wave_height_m
        Grounding Condition: D_eff <= vessel_draft_m

    Args:
        water_depth_m: Charted water column depth in meters (positive down).
        vessel_draft_m: Maximum vessel keel draft in meters.
        tide_height_m: Tidal elevation offset in meters relative to MSL/MLLW.
        wave_height_m: Significant wave height (H_s) in meters (accounts for wave trough plunge).

    Returns:
        Tuple of:
            - is_grounded (bool)
            - effective_depth_m (float)
            - cause (str: 'none', 'landfall', 'shallow_shoal', 'reef_collision_wave_trough')
    """
    if vessel_draft_m <= 0:
        raise ValueError(f"vessel_draft_m must be strictly positive, got {vessel_draft_m}")

    # Effective depth under wave trough plunge
    wave_trough_plunge = 0.5 * max(0.0, wave_height_m)
    effective_depth_m = water_depth_m + tide_height_m - wave_trough_plunge

    if effective_depth_m > vessel_draft_m:
        return False, effective_depth_m, "none"

    # Diagnose physical cause of grounding
    if water_depth_m <= 0.0:
        cause = "landfall"
    elif (water_depth_m + tide_height_m) <= vessel_draft_m:
        cause = "shallow_shoal"
    else:
        cause = "reef_collision_wave_trough"

    return True, effective_depth_m, cause


def check_particle_grounding(
    lat: float,
    lon: float,
    vessel_draft_m: float,
    bathymetry_source: Union[BathymetricGrid, Callable[[float, float], float], float],
    tide_height_m: float = 0.0,
    wave_height_m: float = 0.0,
    timestamp_s: Optional[float] = None,
) -> GroundingEvent:
    """Evaluate particle spatial coordinates against bathymetric depth to detect grounding.

    Args:
        lat: Particle latitude in degrees.
        lon: Particle longitude in degrees.
        vessel_draft_m: Keel draft in meters.
        bathymetry_source: Either a BathymetricGrid instance, a callable (lat, lon) -> depth_m,
            or a scalar float depth.
        tide_height_m: Tidal height offset in meters.
        wave_height_m: Significant wave height (H_s) in meters.
        timestamp_s: Optional simulation timestamp in seconds.

    Returns:
        GroundingEvent instance.
    """
    if isinstance(bathymetry_source, BathymetricGrid):
        water_depth_m = bathymetry_source.sample_depth_bilinear(lat, lon)
    elif callable(bathymetry_source):
        water_depth_m = float(bathymetry_source(lat, lon))
    else:
        water_depth_m = float(bathymetry_source)

    is_grounded, eff_depth, cause = evaluate_grounding_condition(
        water_depth_m=water_depth_m,
        vessel_draft_m=vessel_draft_m,
        tide_height_m=tide_height_m,
        wave_height_m=wave_height_m,
    )

    return GroundingEvent(
        is_grounded=is_grounded,
        grounding_lat=lat,
        grounding_lon=lon,
        water_depth_m=water_depth_m,
        effective_depth_m=eff_depth,
        vessel_draft_m=vessel_draft_m,
        cause=cause,
        timestamp_s=timestamp_s,
    )
