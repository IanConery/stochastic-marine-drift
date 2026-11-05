"""Forward Monte Carlo Lagrangian drift engine and crew behavioral state machine."""

from dataclasses import dataclass, field
from enum import Enum
import json
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from src.agents.schemas import EpistemicPrior
from src.models.debris_field import DebrisField, estimate_debris_field, meters_to_lat_lon_offsets
from src.modules.collision import BathymetricGrid, GroundingEvent, check_particle_grounding
from src.modules.current import compute_total_surface_drift_current
from src.modules.deep_water import simulate_3d_descent
from src.modules.extreme_event import TropicalCyclone
from src.modules.stokes import compute_stokes_drift_vector
from src.modules.wind import perturb_wind_vector, wind_speed_and_direction
from src.profiles.presets import SPANISH_GALLEON_1620
from src.profiles.vessel_class import VesselProfile

# Geographic constants
METERS_PER_DEGREE_LAT = 111139.0


class ParticleState(str, Enum):
    """Dynamic operational and casualty states for a maritime vessel in storm drift."""

    SAILING_DAMAGED = "sailing_damaged"
    HEAVED_TO = "heaved_to"
    JETTISONING_CARGO = "jettisoning_cargo"
    DERELICT_DEAD_DRIFT = "derelict_dead_drift"
    GROUNDED_ON_REEF = "grounded_on_reef"
    FOUNDERED = "foundered"


@dataclass
class DriftParticle:
    """Individual particle within the Monte Carlo ensemble."""

    particle_id: int
    lat: float
    lon: float
    state: ParticleState = ParticleState.DERELICT_DEAD_DRIFT
    jibe_sign: int = 1  # +1 for starboard crosswind leeway, -1 for port
    cargo_mass_kg: float = 60000.0
    buoyancy_factor: float = 0.35
    cumulative_stress: float = 0.0
    stress_threshold: float = 100.0
    trajectory: List[Tuple[float, float, float]] = field(default_factory=list)  # (t_sec, lat, lon)
    terminal_info: Dict[str, Any] = field(default_factory=dict)
    seafloor_lat: Optional[float] = None
    seafloor_lon: Optional[float] = None
    debris_field: Optional[DebrisField] = None

    @property
    def is_terminal(self) -> bool:
        """Return True if particle has ceased surface motion."""
        return self.state in (ParticleState.GROUNDED_ON_REEF, ParticleState.FOUNDERED)


@dataclass
class EnvironmentalConditions:
    """Snapshot or dynamic callback provider for environmental forcing fields."""

    u10_wind_m_s: float = 12.0
    v10_wind_m_s: float = -6.0
    u_ocean_current_m_s: float = 0.25
    v_ocean_current_m_s: float = -0.10
    wave_height_hs_m: float = 3.5
    wave_period_tp_s: float = 8.0
    wave_direction_deg: float = 45.0
    water_depth_m: float = 800.0
    tide_height_m: float = 0.0
    bathymetry_source: Optional[Union[BathymetricGrid, Callable[[float, float], float]]] = None
    active_cyclone: Optional[TropicalCyclone] = None
    deep_current_u_m_s: float = 0.05
    deep_current_v_m_s: float = -0.02


@dataclass
class ForwardSimulationResult:
    """Encapsulates execution histories, terminal distributions, and GeoJSON outputs."""

    particles: List[DriftParticle]
    profile: VesselProfile
    simulation_duration_hours: float
    dt_seconds: float

    @property
    def total_count(self) -> int:
        return len(self.particles)

    @property
    def grounded_particles(self) -> List[DriftParticle]:
        return [p for p in self.particles if p.state == ParticleState.GROUNDED_ON_REEF]

    @property
    def foundered_particles(self) -> List[DriftParticle]:
        return [p for p in self.particles if p.state == ParticleState.FOUNDERED]

    @property
    def active_particles(self) -> List[DriftParticle]:
        return [p for p in self.particles if not p.is_terminal]

    def summary(self) -> Dict[str, Any]:
        """Generate high-level statistical summary of simulation outcomes."""
        grounded_pct = (len(self.grounded_particles) / self.total_count) * 100.0
        foundered_pct = (len(self.foundered_particles) / self.total_count) * 100.0
        active_pct = (len(self.active_particles) / self.total_count) * 100.0

        seafloor_lats = [
            p.seafloor_lat or p.lat
            for p in self.particles
            if p.is_terminal
        ]
        seafloor_lons = [
            p.seafloor_lon or p.lon
            for p in self.particles
            if p.is_terminal
        ]

        mean_resting_lat = float(np.mean(seafloor_lats)) if seafloor_lats else None
        mean_resting_lon = float(np.mean(seafloor_lons)) if seafloor_lons else None

        return {
            "total_particles": self.total_count,
            "duration_hours": self.simulation_duration_hours,
            "vessel_class": self.profile.class_name,
            "grounded_count": len(self.grounded_particles),
            "grounded_percentage": round(grounded_pct, 1),
            "foundered_count": len(self.foundered_particles),
            "foundered_percentage": round(foundered_pct, 1),
            "active_surviving_count": len(self.active_particles),
            "active_percentage": round(active_pct, 1),
            "mean_resting_lat": round(mean_resting_lat, 5) if mean_resting_lat else None,
            "mean_resting_lon": round(mean_resting_lon, 5) if mean_resting_lon else None,
        }

    def to_geojson(self) -> Dict[str, Any]:
        """Export all terminal deposition sites and 95% debris ellipses as a GeoJSON FeatureCollection."""
        features: List[Dict[str, Any]] = []

        for p in self.particles:
            if p.state == ParticleState.GROUNDED_ON_REEF:
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [round(p.lon, 6), round(p.lat, 6)],
                    },
                    "properties": {
                        "particle_id": p.particle_id,
                        "state": p.state.value,
                        "terminal_type": "bathymetric_reef_grounding",
                        **p.terminal_info,
                    },
                })
            elif p.state == ParticleState.FOUNDERED:
                # Add seafloor touchdown point
                dest_lon = p.seafloor_lon or p.lon
                dest_lat = p.seafloor_lat or p.lat
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [round(dest_lon, 6), round(dest_lat, 6)],
                    },
                    "properties": {
                        "particle_id": p.particle_id,
                        "state": p.state.value,
                        "terminal_type": "deep_water_foundering",
                        **p.terminal_info,
                    },
                })
                # Add debris field contour polygon if available
                if p.debris_field is not None:
                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [p.debris_field.get_contour_coordinates("ballast", 0.95)],
                        },
                        "properties": {
                            "particle_id": p.particle_id,
                            "feature_type": "ballast_mound_95_percent_contour",
                        },
                    })

        return {"type": "FeatureCollection", "features": features}

    def to_geojson_str(self, indent: int = 2) -> str:
        """Export GeoJSON as string."""
        return json.dumps(self.to_geojson(), indent=indent)

    def compute_probability_density_grid(
        self, grid_resolution_deg: float = 0.02
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a 2D probability density histogram of terminal wreck positions.

        Args:
            grid_resolution_deg: Spatial bin size in degrees (default ~2 km).

        Returns:
            Tuple of (lat_bins, lon_bins, normalized_density_matrix).
        """
        lats = [p.seafloor_lat or p.lat for p in self.particles if p.is_terminal]
        lons = [p.seafloor_lon or p.lon for p in self.particles if p.is_terminal]

        if not lats:
            return np.array([]), np.array([]), np.array([[]])

        lat_min = math.floor(min(lats) * 10) / 10.0 - 0.05
        lat_max = math.ceil(max(lats) * 10) / 10.0 + 0.05
        lon_min = math.floor(min(lons) * 10) / 10.0 - 0.05
        lon_max = math.ceil(max(lons) * 10) / 10.0 + 0.05

        lat_bins = np.arange(lat_min, lat_max + grid_resolution_deg, grid_resolution_deg)
        lon_bins = np.arange(lon_min, lon_max + grid_resolution_deg, grid_resolution_deg)

        hist, _, _ = np.histogram2d(lats, lons, bins=[lat_bins, lon_bins])
        total = np.sum(hist)
        density = hist / total if total > 0 else hist
        return lat_bins, lon_bins, density


class ForwardDriftSimulator:
    """Ensemble Monte Carlo forward drift simulator."""

    def __init__(
        self,
        profile: VesselProfile = SPANISH_GALLEON_1620,
        prior: Optional[EpistemicPrior] = None,
        survival_model: Optional[Any] = None,
        dt_seconds: float = 600.0,
        num_particles: int = 100,
        enable_crew_behavior: bool = True,
        rng_seed: Optional[int] = 42,
    ) -> None:
        self.profile = profile
        self.prior = prior
        self.survival_model = survival_model
        self.dt_seconds = dt_seconds
        self.num_particles = num_particles
        self.enable_crew_behavior = enable_crew_behavior
        self.rng = np.random.default_rng(rng_seed)

    def _initialize_particles(
        self, lkp_lat: float, lkp_lon: float
    ) -> List[DriftParticle]:
        """Seed initial particle swarm around LKP using epistemic prior covariance."""
        particles: List[DriftParticle] = []

        for pid in range(self.num_particles):
            if self.prior is not None:
                p_lat, p_lon = self.prior.sample_initial_position(lkp_lat, lkp_lon, rng=self.rng)
            else:
                # Default 2-nautical-mile perturbation
                d_lat, d_lon = meters_to_lat_lon_offsets(
                    float(self.rng.normal(0.0, 3704.0)),
                    float(self.rng.normal(0.0, 3704.0)),
                    lkp_lat,
                )
                p_lat, p_lon = lkp_lat + d_lat, lkp_lon + d_lon

            # Random initial jibe sign (starboard +1 vs port -1)
            jibe = 1 if self.rng.random() > 0.5 else -1

            # Base stress threshold with stochastic variability (+-20%)
            base_threshold = 100.0 / self.profile.structural_fatigue_rate
            threshold = base_threshold * float(self.rng.uniform(0.8, 1.2))

            initial_state = (
                ParticleState.SAILING_DAMAGED
                if self.enable_crew_behavior
                else ParticleState.DERELICT_DEAD_DRIFT
            )

            p = DriftParticle(
                particle_id=pid,
                lat=p_lat,
                lon=p_lon,
                state=initial_state,
                jibe_sign=jibe,
                cargo_mass_kg=60000.0,
                buoyancy_factor=0.35,
                cumulative_stress=0.0,
                stress_threshold=threshold,
                trajectory=[(0.0, p_lat, p_lon)],
            )
            particles.append(p)

        return particles

    def _compute_leeway_vector(
        self,
        u_wind: float,
        v_wind: float,
        particle: DriftParticle,
    ) -> Tuple[float, float]:
        """Compute downwind and crosswind leeway velocity vector based on rig and crew state."""
        wind_speed, wind_dir_to = wind_speed_and_direction(u_wind, v_wind)
        if wind_speed == 0.0:
            return 0.0, 0.0

        # Downwind leeway component (DWL)
        dwl_slope = self.profile.downwind_leeway_slope
        dwl_offset = self.profile.downwind_leeway_offset

        # State machine adjustments to leeway:
        # If HEAVED_TO, crew heads vessel into wind; downwind leeway drops by ~50%
        if particle.state == ParticleState.HEAVED_TO:
            dwl_slope *= 0.50
        elif particle.state == ParticleState.SAILING_DAMAGED:
            # Under jury rig, ship is partially steered
            dwl_slope *= 0.75

        dwl_speed = dwl_slope * wind_speed + dwl_offset

        # Crosswind leeway component (CWL)
        cwl_slope = self.profile.crosswind_leeway_slope
        cwl_offset = self.profile.crosswind_leeway_offset
        divergence_deg = self.profile.crosswind_divergence_deg

        if particle.state == ParticleState.HEAVED_TO:
            divergence_deg *= 0.5

        if cwl_slope > 0.0:
            cwl_speed = cwl_slope * wind_speed + cwl_offset
        else:
            cwl_speed = dwl_speed * math.tan(math.radians(divergence_deg))

        # Rotate relative to wind heading
        leeway_heading_deg = (wind_dir_to + particle.jibe_sign * divergence_deg) % 360.0
        leeway_rad = math.radians(leeway_heading_deg)
        total_leeway_speed = math.hypot(dwl_speed, cwl_speed)

        u_leeway = total_leeway_speed * math.sin(leeway_rad)
        v_leeway = total_leeway_speed * math.cos(leeway_rad)

        return u_leeway, v_leeway

    def _update_crew_state_machine(
        self,
        particle: DriftParticle,
        wind_speed: float,
        wave_height: float,
        elapsed_seconds: float,
    ) -> None:
        """Update crew behavioral transitions under severe storm conditions."""
        if not self.enable_crew_behavior or particle.is_terminal:
            return

        # 1. SAILING_DAMAGED -> HEAVED_TO: Triggered if wind > 20 m/s or wave > 4.5m
        if particle.state == ParticleState.SAILING_DAMAGED:
            if wind_speed > 20.0 or wave_height > 4.5:
                particle.state = ParticleState.HEAVED_TO

        # 2. HEAVED_TO -> JETTISONING_CARGO: Water ingress exceeds pumps (stress > 45%)
        elif particle.state == ParticleState.HEAVED_TO:
            if particle.cumulative_stress > (particle.stress_threshold * 0.45):
                particle.state = ParticleState.JETTISONING_CARGO
                # Jettison heavy cargo: reduces cargo mass by 40%, increases reserve buoyancy
                particle.cargo_mass_kg *= 0.60
                particle.buoyancy_factor = min(0.60, particle.buoyancy_factor + 0.15)

        # 3. JETTISONING_CARGO -> DERELICT_DEAD_DRIFT: Steering lost or hull severely compromised
        elif particle.state == ParticleState.JETTISONING_CARGO:
            if particle.cumulative_stress > (particle.stress_threshold * 0.70):
                particle.state = ParticleState.DERELICT_DEAD_DRIFT

    def run(
        self,
        lkp_lat: float,
        lkp_lon: float,
        total_hours: float,
        env: EnvironmentalConditions,
    ) -> ForwardSimulationResult:
        """Execute ensemble Monte Carlo simulation forward in time.

        Args:
            lkp_lat: Initial Last Known Position latitude.
            lkp_lon: Initial Last Known Position longitude.
            total_hours: Total simulation duration in hours.
            env: Environmental forcing conditions (static or dynamic callbacks).

        Returns:
            ForwardSimulationResult instance.
        """
        particles = self._initialize_particles(lkp_lat, lkp_lon)
        total_seconds = total_hours * 3600.0
        num_steps = int(total_seconds / self.dt_seconds)

        for step in range(num_steps):
            current_t = step * self.dt_seconds

            for p in particles:
                if p.is_terminal:
                    continue

                # 1. Environmental Winds & Hurricane Check
                u_wind = env.u10_wind_m_s
                v_wind = env.v10_wind_m_s
                if env.active_cyclone is not None:
                    u_cyc, v_cyc, _ = env.active_cyclone.evaluate_at_point(p.lat, p.lon)
                    u_wind += u_cyc
                    v_wind += v_cyc

                # Apply stochastic Weibull gust turbulence
                u_wind_pert, v_wind_pert = perturb_wind_vector(
                    u10=u_wind,
                    v10=v_wind,
                    rng=self.rng,
                )
                wind_speed = math.hypot(u_wind_pert, v_wind_pert)

                # 2. Waves and Stokes Drift
                hs = env.wave_height_hs_m
                tp = env.wave_period_tp_s
                u_stokes, v_stokes = compute_stokes_drift_vector(
                    wave_height_m=hs,
                    peak_period_s=tp,
                    wave_direction_deg=env.wave_direction_deg,
                )

                # 3. Surface Currents & Ekman Deflection
                u_curr, v_curr = compute_total_surface_drift_current(
                    u_ocean_geostrophic=env.u_ocean_current_m_s,
                    v_ocean_geostrophic=env.v_ocean_current_m_s,
                    u10_wind=u_wind_pert,
                    v10_wind=v_wind_pert,
                    latitude_deg=p.lat,
                    include_ekman=True,
                )

                # 4. Leeway Drift & Jibe Switching
                # Jibe check: P(jibe) = 1 - exp(-lambda * dt)
                jibe_rate = self.profile.jibe_probability_per_hour / 3600.0
                if self.rng.random() < (1.0 - math.exp(-jibe_rate * self.dt_seconds)):
                    p.jibe_sign *= -1

                u_leeway, v_leeway = self._compute_leeway_vector(u_wind_pert, v_wind_pert, p)

                # 5. Total Particle Velocity & Spatial Step
                v_total_x = u_curr + u_stokes + u_leeway
                v_total_y = v_curr + v_stokes + v_leeway

                dx_m = v_total_x * self.dt_seconds
                dy_m = v_total_y * self.dt_seconds

                d_lat, d_lon = meters_to_lat_lon_offsets(dx_m, dy_m, p.lat)
                p.lat += d_lat
                p.lon += d_lon
                p.trajectory.append((current_t + self.dt_seconds, p.lat, p.lon))

                # 6. Bathymetric Reef Collision Check
                bathymetry_source = env.bathymetry_source if env.bathymetry_source is not None else env.water_depth_m
                grounding_event = check_particle_grounding(
                    lat=p.lat,
                    lon=p.lon,
                    vessel_draft_m=self.profile.draft_meters,
                    bathymetry_source=bathymetry_source,
                    tide_height_m=env.tide_height_m,
                    wave_height_m=hs,
                    timestamp_s=current_t + self.dt_seconds,
                )

                if grounding_event.is_grounded:
                    p.state = ParticleState.GROUNDED_ON_REEF
                    p.terminal_info = {
                        "cause": grounding_event.cause,
                        "water_depth_m": grounding_event.water_depth_m,
                        "effective_depth_m": grounding_event.effective_depth_m,
                        "time_hours": round((current_t + self.dt_seconds) / 3600.0, 2),
                    }
                    continue

                # 7. Dynamic Hull Stress Accumulation & Crew State Machine
                wave_steepness = hs / max(1.0, 9.80665 * (tp**2))
                stress_rate = (
                    (wind_speed / 15.0) ** 1.5
                    + (hs / 3.0) ** 2.0
                    + (wave_steepness * 50.0)
                ) * self.profile.structural_fatigue_rate

                p.cumulative_stress += stress_rate * (self.dt_seconds / 3600.0)
                self._update_crew_state_machine(p, wind_speed, hs, current_t)

                # 8. Foundering Check: Dynamic Survival Model or Structural Stress Threshold
                should_founder = False
                if self.survival_model is not None:
                    macro_val = getattr(self.profile.macro_archetype, "value", str(self.profile.macro_archetype))
                    step_hazard = self.survival_model.predict_instantaneous_hazard(
                        hs_m=hs,
                        tp_s=tp,
                        wind_speed_ms=wind_speed,
                        freeboard_m=self.profile.freeboard_meters * p.buoyancy_factor,
                        cumulative_stress=p.cumulative_stress,
                        vessel_subtype=self.profile.class_name,
                        macro_archetype=macro_val,
                        dt_seconds=self.dt_seconds,
                    )
                    if self.rng.random() < step_hazard:
                        should_founder = True
                        p.terminal_info["survival_hazard_step"] = round(step_hazard, 6)
                else:
                    if p.cumulative_stress >= p.stress_threshold:
                        should_founder = True

                if should_founder:
                    p.state = ParticleState.FOUNDERED
                    water_depth = grounding_event.water_depth_m

                    # Execute 3D water column descent down to seafloor
                    final_x_m, final_y_m, descent_time_s, _ = simulate_3d_descent(
                        sps_x_m=0.0,
                        sps_y_m=0.0,
                        seafloor_depth_m=max(10.0, water_depth),
                        profile=self.profile,
                        current_field_fn=lambda x, y, z: (env.deep_current_u_m_s, env.deep_current_v_m_s),
                    )

                    d_seafloor_lat, d_seafloor_lon = meters_to_lat_lon_offsets(final_x_m, final_y_m, p.lat)
                    p.seafloor_lat = p.lat + d_seafloor_lat
                    p.seafloor_lon = p.lon + d_seafloor_lon

                    # Compute archaeological debris field
                    p.debris_field = estimate_debris_field(
                        touchdown_lat=p.seafloor_lat,
                        touchdown_lon=p.seafloor_lon,
                        deep_current_u_m_s=env.deep_current_u_m_s,
                        deep_current_v_m_s=env.deep_current_v_m_s,
                        descent_duration_s=descent_time_s,
                        profile=self.profile,
                    )

                    p.terminal_info = {
                        "cause": "dynamic_hull_stress_failure",
                        "sinking_depth_m": water_depth,
                        "descent_time_seconds": round(descent_time_s, 1),
                        "time_hours": round((current_t + self.dt_seconds) / 3600.0, 2),
                    }
                    continue

        return ForwardSimulationResult(
            particles=particles,
            profile=self.profile,
            simulation_duration_hours=total_hours,
            dt_seconds=self.dt_seconds,
        )
