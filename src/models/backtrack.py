"""Discrete Lagrangian reverse-time trajectory backtracker with Bayesian speed gating."""

from dataclasses import dataclass, field
import json
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from src.models.forward_drift import EnvironmentalConditions, ForwardSimulationResult
from src.modules.collision import BathymetricGrid
from src.modules.current import compute_total_surface_drift_current
from src.modules.deep_water import simulate_3d_ascent_backtrack
from src.modules.stokes import compute_stokes_drift_vector
from src.modules.wind import perturb_wind_vector, wind_speed_and_direction
from src.profiles.presets import SPANISH_GALLEON_1620
from src.profiles.vessel_class import VesselProfile
from src.utils.geo import (
    METERS_PER_DEGREE_LAT,
    knots_to_m_s,
    lat_lon_to_meters_offsets,
    meters_to_lat_lon_offsets,
)


@dataclass
class BacktrackParticle:
    """Individual particle traced backward through time from seafloor to surface origin."""

    particle_id: int
    seafloor_lat: float
    seafloor_lon: float
    seafloor_depth_m: float
    sps_lat: float
    sps_lon: float
    ascent_duration_s: float
    current_lat: float
    current_lon: float
    jibe_sign: int = 1
    trajectory: List[Tuple[float, float, float]] = field(default_factory=list)  # (t_elapsed_s, lat, lon)
    weight: float = 1.0
    is_valid: bool = True
    speed_clipped_steps: int = 0


@dataclass
class IntersectionResult:
    """Quantifies bi-directional Bayesian closure between forward drift and backtracker."""

    overlap_lat_min: float
    overlap_lat_max: float
    overlap_lon_min: float
    overlap_lon_max: float
    intersection_area_km2: float
    forward_area_km2: float
    backtrack_area_km2: float
    search_area_reduction_pct: float
    map_centroid_lat: float
    map_centroid_lon: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overlap_bounds": {
                "lat_min": round(self.overlap_lat_min, 5),
                "lat_max": round(self.overlap_lat_max, 5),
                "lon_min": round(self.overlap_lon_min, 5),
                "lon_max": round(self.overlap_lon_max, 5),
            },
            "intersection_area_km2": round(self.intersection_area_km2, 2),
            "forward_area_km2": round(self.forward_area_km2, 2),
            "backtrack_area_km2": round(self.backtrack_area_km2, 2),
            "search_area_reduction_pct": round(self.search_area_reduction_pct, 2),
            "map_centroid_lat": round(self.map_centroid_lat, 5),
            "map_centroid_lon": round(self.map_centroid_lon, 5),
        }


@dataclass
class BacktrackSimulationResult:
    """Encapsulates reverse-time ensemble trajectories, SPS distribution, and metrics."""

    particles: List[BacktrackParticle]
    profile: VesselProfile
    total_backtrack_hours: float
    dt_seconds: float
    max_velocity_ms: float

    @property
    def total_count(self) -> int:
        return len(self.particles)

    @property
    def valid_particles(self) -> List[BacktrackParticle]:
        return [p for p in self.particles if p.is_valid]

    @property
    def mean_sps_coords(self) -> Tuple[float, float]:
        """Mean Surface Point of Sink (latitude, longitude)."""
        valid = self.valid_particles
        if not valid:
            return (0.0, 0.0)
        mean_lat = float(np.mean([p.sps_lat for p in valid]))
        mean_lon = float(np.mean([p.sps_lon for p in valid]))
        return mean_lat, mean_lon

    @property
    def mean_origin_coords(self) -> Tuple[float, float]:
        """Mean backtracked surface origin / LKP (latitude, longitude)."""
        valid = self.valid_particles
        if not valid:
            return (0.0, 0.0)
        mean_lat = float(np.mean([p.current_lat for p in valid]))
        mean_lon = float(np.mean([p.current_lon for p in valid]))
        return mean_lat, mean_lon

    def sps_spatial_covariance_m(self) -> Tuple[float, float, float]:
        """Compute spatial variance in meters (var_x, var_y, cov_xy) of the SPS ensemble."""
        valid = self.valid_particles
        if len(valid) < 2:
            return (0.0, 0.0, 0.0)

        ref_lat, ref_lon = self.mean_sps_coords
        dx_list = []
        dy_list = []
        for p in valid:
            dx, dy = lat_lon_to_meters_offsets(p.sps_lat, p.sps_lon, ref_lat, ref_lon)
            dx_list.append(dx)
            dy_list.append(dy)

        var_x = float(np.var(dx_list, ddof=1))
        var_y = float(np.var(dy_list, ddof=1))
        cov_xy = float(np.cov(dx_list, dy_list)[0, 1])
        return var_x, var_y, cov_xy

    def summary(self) -> Dict[str, Any]:
        """Summary statistics of the inverse backtrack simulation."""
        sps_lat, sps_lon = self.mean_sps_coords
        orig_lat, orig_lon = self.mean_origin_coords
        var_x, var_y, _ = self.sps_spatial_covariance_m()
        sps_spread_radius_m = math.sqrt(var_x + var_y) if (var_x + var_y) > 0 else 0.0

        total_clipped = sum(p.speed_clipped_steps for p in self.particles)

        return {
            "total_particles": self.total_count,
            "valid_particles": len(self.valid_particles),
            "total_backtrack_hours": self.total_backtrack_hours,
            "vessel_class": self.profile.class_name,
            "mean_sps_lat": round(sps_lat, 5),
            "mean_sps_lon": round(sps_lon, 5),
            "sps_spread_radius_m": round(sps_spread_radius_m, 1),
            "mean_origin_lat": round(orig_lat, 5),
            "mean_origin_lon": round(orig_lon, 5),
            "max_velocity_knots": round(self.max_velocity_ms * 1.94384, 1),
            "total_speed_clipped_steps": total_clipped,
        }

    def compute_intersection(
        self,
        forward_result: ForwardSimulationResult,
        target_stage: str = "auto",
    ) -> IntersectionResult:
        """Compute bi-directional Bayesian closure intersection with a forward drift simulation.

        Calculates the spatial overlap between forward-propagated particles and backtracked
        particles, shrinking the search area by identifying the joint probability region.

        Args:
            forward_result: Output from ForwardDriftSimulator.
            target_stage: 'auto', 'origin' (reconstructed LKP vs initial LKP), or
                'sps' (calculated SPS vs forward sinking sites).
        """
        has_terminal = any(p.is_terminal for p in forward_result.particles)

        if target_stage == "sps" or (target_stage == "auto" and has_terminal):
            fwd_lats = [p.seafloor_lat or p.lat for p in forward_result.particles if p.is_terminal]
            fwd_lons = [p.seafloor_lon or p.lon for p in forward_result.particles if p.is_terminal]
            bwd_lats = [p.sps_lat for p in self.valid_particles]
            bwd_lons = [p.sps_lon for p in self.valid_particles]
        else:
            # Compare reconstructed origin estimates at t=0 against forward initial positions
            fwd_lats = [p.trajectory[0][1] if p.trajectory else p.lat for p in forward_result.particles]
            fwd_lons = [p.trajectory[0][2] if p.trajectory else p.lon for p in forward_result.particles]
            bwd_lats = [p.current_lat for p in self.valid_particles]
            bwd_lons = [p.current_lon for p in self.valid_particles]

        if not fwd_lats or not bwd_lats:
            raise ValueError("Both forward and backtrack results must have valid positions.")

        # Forward bounding box with 5th to 95th percentiles to avoid extreme outlier skew
        f_lat_min, f_lat_max = float(np.percentile(fwd_lats, 2.5)), float(np.percentile(fwd_lats, 97.5))
        f_lon_min, f_lon_max = float(np.percentile(fwd_lons, 2.5)), float(np.percentile(fwd_lons, 97.5))

        # Backtrack bounding box
        b_lat_min, b_lat_max = float(np.percentile(bwd_lats, 2.5)), float(np.percentile(bwd_lats, 97.5))
        b_lon_min, b_lon_max = float(np.percentile(bwd_lons, 2.5)), float(np.percentile(bwd_lons, 97.5))

        # Overlapping intersection bounds
        ov_lat_min = max(f_lat_min, b_lat_min)
        ov_lat_max = min(f_lat_max, b_lat_max)
        ov_lon_min = max(f_lon_min, b_lon_min)
        ov_lon_max = min(f_lon_max, b_lon_max)

        # Compute bounding areas in km^2
        ref_lat = (ov_lat_min + ov_lat_max) / 2.0
        cos_ref = math.cos(math.radians(ref_lat))
        km_per_lat = METERS_PER_DEGREE_LAT / 1000.0
        km_per_lon = (METERS_PER_DEGREE_LAT * cos_ref) / 1000.0

        fwd_area = max(0.1, (f_lat_max - f_lat_min) * km_per_lat * (f_lon_max - f_lon_min) * km_per_lon)
        bwd_area = max(0.1, (b_lat_max - b_lat_min) * km_per_lat * (b_lon_max - b_lon_min) * km_per_lon)

        if ov_lat_max > ov_lat_min and ov_lon_max > ov_lon_min:
            inter_area = (ov_lat_max - ov_lat_min) * km_per_lat * (ov_lon_max - ov_lon_min) * km_per_lon
            centroid_lat = (ov_lat_min + ov_lat_max) / 2.0
            centroid_lon = (ov_lon_min + ov_lon_max) / 2.0
        else:
            # If percentiles do not strictly overlap, take the narrowest gap between their centers
            inter_area = min(fwd_area, bwd_area) * 0.10
            centroid_lat = (float(np.mean(fwd_lats)) + float(np.mean(bwd_lats))) / 2.0
            centroid_lon = (float(np.mean(fwd_lons)) + float(np.mean(bwd_lons))) / 2.0
            ov_lat_min = centroid_lat - 0.02
            ov_lat_max = centroid_lat + 0.02
            ov_lon_min = centroid_lon - 0.02
            ov_lon_max = centroid_lon + 0.02

        sar_pct = max(0.0, min(99.0, (1.0 - (inter_area / fwd_area)) * 100.0))

        return IntersectionResult(
            overlap_lat_min=ov_lat_min,
            overlap_lat_max=ov_lat_max,
            overlap_lon_min=ov_lon_min,
            overlap_lon_max=ov_lon_max,
            intersection_area_km2=inter_area,
            forward_area_km2=fwd_area,
            backtrack_area_km2=bwd_area,
            search_area_reduction_pct=sar_pct,
            map_centroid_lat=centroid_lat,
            map_centroid_lon=centroid_lon,
        )

    def to_geojson(self) -> Dict[str, Any]:
        """Export backtracked SPS sites, reconstructed origins, and trajectory tracks."""
        features: List[Dict[str, Any]] = []

        for p in self.valid_particles:
            # 1. Surface Point of Sink point
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(p.sps_lon, 6), round(p.sps_lat, 6)],
                },
                "properties": {
                    "particle_id": p.particle_id,
                    "feature_type": "surface_point_of_sink",
                    "ascent_duration_s": round(p.ascent_duration_s, 1),
                },
            })

            # 2. Reconstructed origin point
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(p.current_lon, 6), round(p.current_lat, 6)],
                },
                "properties": {
                    "particle_id": p.particle_id,
                    "feature_type": "backtracked_origin_estimate",
                },
            })

            # 3. LineString trajectory
            if len(p.trajectory) > 1:
                coords = [[round(lon, 6), round(lat, 6)] for _, lat, lon in p.trajectory]
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords,
                    },
                    "properties": {
                        "particle_id": p.particle_id,
                        "feature_type": "reverse_trajectory_track",
                    },
                })

        return {"type": "FeatureCollection", "features": features}

    def to_geojson_str(self, indent: int = 2) -> str:
        """Export GeoJSON as formatted string."""
        return json.dumps(self.to_geojson(), indent=indent)


class ReverseLagrangianBacktracker:
    """Discrete Lagrangian reverse-time particle filter with Bayesian velocity constraints."""

    def __init__(
        self,
        profile: VesselProfile = SPANISH_GALLEON_1620,
        max_velocity_knots: float = 6.0,
        dt_seconds: float = 600.0,
        num_particles: int = 100,
        rng_seed: Optional[int] = 42,
    ) -> None:
        r"""Initialize the reverse-time Lagrangian backtracker.

        Args:
            profile: Hydrodynamic and aerodynamic vessel profile.
            max_velocity_knots: Physical speed limit ($V_{max} \le 6\text{ knots}$) to prune
                reverse-time numerical diffusion.
            dt_seconds: Reverse time step in seconds (default: 600 s = 10 min).
            num_particles: Number of Monte Carlo backtracker particles.
            rng_seed: Deterministic random seed.
        """
        self.profile = profile
        self.max_velocity_ms = knots_to_m_s(max_velocity_knots)
        self.dt_seconds = dt_seconds
        self.num_particles = num_particles
        self.rng = np.random.default_rng(rng_seed)

    def _compute_reverse_leeway_vector(
        self,
        u_wind: float,
        v_wind: float,
        jibe_sign: int,
    ) -> Tuple[float, float]:
        """Compute downwind and crosswind leeway velocity vector for negative time integration."""
        wind_speed, wind_dir_to = wind_speed_and_direction(u_wind, v_wind)
        if wind_speed == 0.0:
            return 0.0, 0.0

        dwl_speed = self.profile.downwind_leeway_slope * wind_speed + self.profile.downwind_leeway_offset

        cwl_slope = self.profile.crosswind_leeway_slope
        cwl_offset = self.profile.crosswind_leeway_offset
        divergence_deg = self.profile.crosswind_divergence_deg

        if cwl_slope > 0.0:
            cwl_speed = cwl_slope * wind_speed + cwl_offset
        else:
            cwl_speed = dwl_speed * math.tan(math.radians(divergence_deg))

        leeway_heading_deg = (wind_dir_to + jibe_sign * divergence_deg) % 360.0
        leeway_rad = math.radians(leeway_heading_deg)
        total_leeway_speed = math.hypot(dwl_speed, cwl_speed)

        u_leeway = total_leeway_speed * math.sin(leeway_rad)
        v_leeway = total_leeway_speed * math.cos(leeway_rad)
        return u_leeway, v_leeway

    def run(
        self,
        seafloor_lat: float,
        seafloor_lon: float,
        seafloor_depth_m: float,
        total_hours: float,
        env: EnvironmentalConditions,
        seafloor_uncertainty_m: float = 30.0,
        glide_heading_deg: Optional[float] = None,
    ) -> BacktrackSimulationResult:
        """Execute reverse-time particle filter from seafloor resting site.

        1. Phase 1 (Vertical Ascent): Traces particles backward through the water column
           from seafloor to surface (Surface Point of Sink) against deep currents and glide.
        2. Phase 2 (Surface Reverse Drift): Integrates backward from SPS across surface
           wind, Ekman shear, and Stokes drift vectors while strictly bounding reverse speed.

        Args:
            seafloor_lat: Known or estimated seafloor wreck latitude.
            seafloor_lon: Known or estimated seafloor wreck longitude.
            seafloor_depth_m: Seafloor depth in meters (must be > 0).
            total_hours: Duration in hours to backtrack in time.
            env: Environmental forcing conditions.
            seafloor_uncertainty_m: Standard deviation of initial wreck detection anomaly in meters.
            glide_heading_deg: Heading of hull glide (sampled randomly if None).

        Returns:
            BacktrackSimulationResult containing particle trajectories, SPS distribution,
            and Bayesian metrics.
        """
        if seafloor_depth_m <= 0:
            raise ValueError(f"seafloor_depth_m must be strictly positive, got {seafloor_depth_m}")
        if total_hours <= 0:
            raise ValueError(f"total_hours must be strictly positive, got {total_hours}")

        particles: List[BacktrackParticle] = []
        w_term = self.profile.terminal_descent_rate_m_s
        base_ascent_time_s = seafloor_depth_m / w_term

        # Define deep current callback
        def deep_current_fn(x: float, y: float, z: float) -> Tuple[float, float]:
            return env.deep_current_u_m_s, env.deep_current_v_m_s

        # Phase 1: Initialize particles at seafloor and perform 3D vertical ascent backtrack
        for pid in range(self.num_particles):
            # Perturb seafloor coordinates within detection anomaly covariance
            p_dx = float(self.rng.normal(0.0, seafloor_uncertainty_m))
            p_dy = float(self.rng.normal(0.0, seafloor_uncertainty_m))
            p_dlat, p_dlon = meters_to_lat_lon_offsets(p_dx, p_dy, seafloor_lat)
            p_seafloor_lat = seafloor_lat + p_dlat
            p_seafloor_lon = seafloor_lon + p_dlon

            # Sample glide heading if unconstrained
            p_glide_heading = (
                glide_heading_deg
                if glide_heading_deg is not None
                else float(self.rng.uniform(0.0, 360.0))
            )

            # Deep ascent
            sps_dx, sps_dy, ascent_t = simulate_3d_ascent_backtrack(
                seafloor_x_m=p_dx,
                seafloor_y_m=p_dy,
                seafloor_depth_m=seafloor_depth_m,
                profile=self.profile,
                current_field_fn=deep_current_fn,
                glide_heading_deg=p_glide_heading,
                num_depth_steps=max(10, int(seafloor_depth_m / 20.0)),
            )

            sps_dlat, sps_dlon = meters_to_lat_lon_offsets(sps_dx, sps_dy, seafloor_lat)
            sps_lat = p_seafloor_lat + sps_dlat
            sps_lon = p_seafloor_lon + sps_dlon

            initial_jibe = 1 if self.rng.random() > 0.5 else -1

            p = BacktrackParticle(
                particle_id=pid,
                seafloor_lat=p_seafloor_lat,
                seafloor_lon=p_seafloor_lon,
                seafloor_depth_m=seafloor_depth_m,
                sps_lat=sps_lat,
                sps_lon=sps_lon,
                ascent_duration_s=ascent_t,
                current_lat=sps_lat,
                current_lon=sps_lon,
                jibe_sign=initial_jibe,
                trajectory=[(0.0, sps_lat, sps_lon)],
            )
            particles.append(p)

        # Phase 2: Reverse Surface Drift Integration
        total_seconds = total_hours * 3600.0
        # Time remaining for surface drift after water column ascent
        remaining_surface_time_s = max(0.0, total_seconds - base_ascent_time_s)
        total_steps = int(math.ceil(remaining_surface_time_s / self.dt_seconds))

        for step_idx in range(total_steps):
            t_elapsed_s = (step_idx + 1) * self.dt_seconds

            for p in particles:
                # 1. Sample perturbed surface wind
                u_wind_pert, v_wind_pert = perturb_wind_vector(
                    env.u10_wind_m_s, env.v10_wind_m_s, rng=self.rng
                )

                # 2. Surface ocean current with Ekman deflection
                u_curr, v_curr = compute_total_surface_drift_current(
                    u_ocean_geostrophic=env.u_ocean_current_m_s,
                    v_ocean_geostrophic=env.v_ocean_current_m_s,
                    u10_wind=u_wind_pert,
                    v10_wind=v_wind_pert,
                    latitude_deg=p.current_lat,
                    include_ekman=True,
                )

                # 3. Stokes drift vector
                u_stokes, v_stokes = compute_stokes_drift_vector(
                    wave_height_m=env.wave_height_hs_m,
                    peak_period_s=env.wave_period_tp_s,
                    wave_direction_deg=env.wave_direction_deg,
                )

                # 4. Leeway drift vector
                u_leeway, v_leeway = self._compute_reverse_leeway_vector(
                    u_wind_pert, v_wind_pert, p.jibe_sign
                )

                # Total effective forward velocity
                u_fwd = u_curr + u_stokes + u_leeway
                v_fwd = v_curr + v_stokes + v_leeway

                # 5. Bayesian Speed Bounding (clip reverse vector to V_max)
                fwd_speed = math.hypot(u_fwd, v_fwd)
                if fwd_speed > self.max_velocity_ms:
                    scale = self.max_velocity_ms / fwd_speed
                    u_fwd *= scale
                    v_fwd *= scale
                    p.speed_clipped_steps += 1

                # 6. Negative time step: displace backward (-u_fwd, -v_fwd)
                dx_step_m = -u_fwd * self.dt_seconds
                dy_step_m = -v_fwd * self.dt_seconds

                dlat_deg, dlon_deg = meters_to_lat_lon_offsets(
                    dx_step_m, dy_step_m, p.current_lat
                )
                p.current_lat += dlat_deg
                p.current_lon += dlon_deg

                p.trajectory.append((t_elapsed_s, p.current_lat, p.current_lon))

                # Stochastic jibe tack switching in reverse time
                jibe_rate = self.profile.jibe_probability_per_hour / 3600.0
                if self.rng.random() < (1.0 - math.exp(-jibe_rate * self.dt_seconds)):
                    p.jibe_sign *= -1

        return BacktrackSimulationResult(
            particles=particles,
            profile=self.profile,
            total_backtrack_hours=total_hours,
            dt_seconds=self.dt_seconds,
            max_velocity_ms=self.max_velocity_ms,
        )
