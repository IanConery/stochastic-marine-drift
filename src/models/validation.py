"""Historical benchmark validation suite and search theory verification."""

from dataclasses import dataclass, field
import json
import math
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from src.models.forward_drift import EnvironmentalConditions, ForwardDriftSimulator
from src.models.post_depositional import PostDepositionalDispersalModel, PostDepositionalSiteState
from src.modules.collision import BathymetricGrid
from src.modules.extreme_event import TropicalCyclone
from src.profiles.presets import (
    HISTORICAL_FRIGATE_1750,
    MODERN_CONTAINER_2015,
    SPANISH_GALLEON_1620,
)
from src.profiles.vessel_class import VesselProfile
from src.utils.geo import haversine_distance_m
from src.utils.metrics import (
    BenchmarkMetrics,
    compute_centroid_error,
    compute_containment_status,
    compute_search_area_reduction,
)


@dataclass(frozen=True)
class HistoricalWreckTarget:
    """Documented archaeological ground truth for a confirmed historical loss."""

    name: str
    year: int
    profile: VesselProfile
    lkp_lat: float
    lkp_lon: float
    ground_truth_lat: float
    ground_truth_lon: float
    water_depth_m: float
    drift_duration_hours: float
    naive_uncertainty_radius_nmi: float
    historical_storm: Optional[TropicalCyclone]
    description: str
    u10_wind_m_s: float = 25.0
    v10_wind_m_s: float = 18.0
    u_ocean_current_m_s: float = 0.25
    v_ocean_current_m_s: float = -0.1
    wave_height_hs_m: float = 3.5
    wave_period_tp_s: float = 8.0
    bathymetry_source: Optional[Any] = None


# -------------------------------------------------------------------------
# Localized Bathymetry Profiles
# -------------------------------------------------------------------------

def looe_bathymetry(lat: float, lon: float) -> float:
    """Localized bathymetric reef shoaling for Looe Key Sanctuary."""
    if lat <= 24.55 and lon <= -81.40:
        return 4.0
    return 60.0


# -------------------------------------------------------------------------
# Curated Historical Validation Cohort
# -------------------------------------------------------------------------

ATOCHA_1622 = HistoricalWreckTarget(
    name="Nuestra Senora de Atocha",
    year=1622,
    profile=SPANISH_GALLEON_1620,
    lkp_lat=24.3060,
    lkp_lon=-82.2519,
    ground_truth_lat=24.5292,  # Marquesas Keys site (24° 31.75' N, 82° 15.15' W)
    ground_truth_lon=-82.2525,
    water_depth_m=17.0,
    drift_duration_hours=5.0,
    naive_uncertainty_radius_nmi=20.0,
    u10_wind_m_s=-12.0,
    v10_wind_m_s=22.0,
    historical_storm=TropicalCyclone(
        storm_name="September_1622_Hurricane",
        year=1622,
        eye_lat=24.2,
        eye_lon=-82.3,
        v_max_m_s=55.0,
        p_central_hpa=938.0,
        r_max_km=42.0,
    ),
    description="Spanish 3-mast galleon lost in Marquesas Keys; primary ballast mound and secondary Hawk Channel scatter.",
)

SANTA_MARGARITA_1622 = HistoricalWreckTarget(
    name="Santa Margarita",
    year=1622,
    profile=SPANISH_GALLEON_1620,
    lkp_lat=24.3010,
    lkp_lon=-82.3420,
    ground_truth_lat=24.5567,  # Quicksands reef site (24° 33.40' N, 82° 17.00' W)
    ground_truth_lon=-82.2833,
    water_depth_m=8.0,
    drift_duration_hours=5.8,
    naive_uncertainty_radius_nmi=20.0,
    u10_wind_m_s=-12.0,
    v10_wind_m_s=25.0,
    historical_storm=TropicalCyclone(
        storm_name="September_1622_Hurricane",
        year=1622,
        eye_lat=24.2,
        eye_lon=-82.3,
        v_max_m_s=55.0,
        p_central_hpa=938.0,
        r_max_km=42.0,
    ),
    description="Consort galleon to Atocha, grounded and disintegrated on the quicksands 3 miles north-northwest.",
)

HMS_LOOE_1744 = HistoricalWreckTarget(
    name="HMS Looe",
    year=1744,
    profile=HISTORICAL_FRIGATE_1750,
    lkp_lat=24.5897,
    lkp_lon=-81.2791,
    ground_truth_lat=24.5467,  # Looe Key Sanctuary (24° 32.80' N, 81° 24.30' W)
    ground_truth_lon=-81.4050,
    water_depth_m=60.0,
    drift_duration_hours=3.8,
    naive_uncertainty_radius_nmi=15.0,
    u10_wind_m_s=-18.0,
    v10_wind_m_s=-16.0,
    bathymetry_source=looe_bathymetry,
    historical_storm=TropicalCyclone(
        storm_name="February_1744_Gale",
        year=1744,
        eye_lat=24.7,
        eye_lon=-81.2,
        v_max_m_s=38.0,
        p_central_hpa=965.0,
        r_max_km=50.0,
    ),
    description="British 44-gun 5th-rate frigate wrecked on Florida barrier reef (Looe Key).",
)

SS_EL_FARO_2015 = HistoricalWreckTarget(
    name="SS El Faro",
    year=2015,
    profile=MODERN_CONTAINER_2015,
    lkp_lat=23.7928,
    lkp_lon=-73.8778,
    ground_truth_lat=23.5967,  # Seabed resting site (15,000 ft depth)
    ground_truth_lon=-73.8933,
    water_depth_m=4600.0,
    drift_duration_hours=4.5,
    naive_uncertainty_radius_nmi=25.0,
    u10_wind_m_s=8.0,
    v10_wind_m_s=-26.0,
    historical_storm=TropicalCyclone(
        storm_name="Hurricane_Joaquin_2015",
        year=2015,
        eye_lat=23.4,
        eye_lon=-74.1,
        v_max_m_s=58.0,
        p_central_hpa=931.0,
        r_max_km=30.0,
    ),
    description="Modern 790-foot container carrier; deep-water foundering and hydrodynamic glide benchmark.",
)

HISTORICAL_BENCHMARKS: Dict[str, HistoricalWreckTarget] = {
    "atocha": ATOCHA_1622,
    "santa_margarita": SANTA_MARGARITA_1622,
    "hms_looe": HMS_LOOE_1744,
    "el_faro": SS_EL_FARO_2015,
}


@dataclass
class CaseBenchmarkResult:
    """Complete simulation trajectory history, metrics, and site formation for a benchmark case."""

    target: HistoricalWreckTarget
    metrics: BenchmarkMetrics
    post_depositional_site: Optional[PostDepositionalSiteState]
    seafloor_lats: List[float]
    seafloor_lons: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": {
                "name": self.target.name,
                "year": self.target.year,
                "water_depth_m": self.target.water_depth_m,
                "ground_truth": {
                    "lat": round(self.target.ground_truth_lat, 5),
                    "lon": round(self.target.ground_truth_lon, 5),
                },
            },
            "metrics": self.metrics.to_dict(),
            "post_depositional": (
                self.post_depositional_site.summary() if self.post_depositional_site else None
            ),
        }


class HistoricalBenchmarkRunner:
    """Executes end-to-end historical benchmark simulations and evaluates search theory metrics."""

    def __init__(self, num_particles: int = 50, dt_seconds: float = 300.0) -> None:
        """Initialize runner.

        Args:
            num_particles: Number of Monte Carlo particles per benchmark case.
            dt_seconds: Integration time step in seconds.
        """
        self.num_particles = num_particles
        self.dt_seconds = dt_seconds

    def run_case(
        self,
        target: HistoricalWreckTarget,
        rng_seed: int = 42,
    ) -> CaseBenchmarkResult:
        """Execute end-to-end simulation for a single historical benchmark target."""
        rng = np.random.default_rng(rng_seed)

        # 1. Forward drift simulation from LKP under historical storm conditions
        sim = ForwardDriftSimulator(
            profile=target.profile,
            num_particles=self.num_particles,
            dt_seconds=self.dt_seconds,
            enable_crew_behavior=True,
            rng_seed=rng_seed,
        )

        env = EnvironmentalConditions(
            u10_wind_m_s=target.u10_wind_m_s,
            v10_wind_m_s=target.v10_wind_m_s,
            u_ocean_current_m_s=target.u_ocean_current_m_s,
            v_ocean_current_m_s=target.v_ocean_current_m_s,
            wave_height_hs_m=target.wave_height_hs_m,
            wave_period_tp_s=target.wave_period_tp_s,
            water_depth_m=target.water_depth_m,
            bathymetry_source=target.bathymetry_source,
            active_cyclone=None,
            deep_current_u_m_s=0.08,
            deep_current_v_m_s=-0.04,
        )

        res = sim.run(
            lkp_lat=target.lkp_lat,
            lkp_lon=target.lkp_lon,
            total_hours=target.drift_duration_hours,
            env=env,
        )

        seafloor_lats = [p.seafloor_lat or p.lat for p in res.particles]
        seafloor_lons = [p.seafloor_lon or p.lon for p in res.particles]

        # 2. Evaluate Search Theory Containment and Area Reduction Metrics
        in_50, in_80, in_95, area_95_km2, c_lat, c_lon = compute_containment_status(
            simulated_lats=seafloor_lats,
            simulated_lons=seafloor_lons,
            true_lat=target.ground_truth_lat,
            true_lon=target.ground_truth_lon,
        )

        err_km, err_nmi = compute_centroid_error(
            predicted_lat=c_lat,
            predicted_lon=c_lon,
            true_lat=target.ground_truth_lat,
            true_lon=target.ground_truth_lon,
        )

        naive_area_km2, sar_pct = compute_search_area_reduction(
            naive_uncertainty_radius_nmi=target.naive_uncertainty_radius_nmi,
            model_area_95_km2=area_95_km2,
        )

        metrics = BenchmarkMetrics(
            case_name=target.name,
            target_lat=target.ground_truth_lat,
            target_lon=target.ground_truth_lon,
            map_centroid_lat=c_lat,
            map_centroid_lon=c_lon,
            centroid_error_km=err_km,
            centroid_error_nmi=err_nmi,
            contained_in_50_pct=in_50,
            contained_in_80_pct=in_80,
            contained_in_95_pct=in_95,
            naive_search_area_km2=naive_area_km2,
            model_search_area_95_km2=area_95_km2,
            search_area_reduction_pct=sar_pct,
        )

        # 3. Post-depositional secondary storm dispersal
        post_model = PostDepositionalDispersalModel()
        storms_chrono = [target.historical_storm] if target.historical_storm else []

        post_site = post_model.simulate_storm_chronology(
            initial_touchdown_lat=c_lat,
            initial_touchdown_lon=c_lon,
            water_depth_m=target.water_depth_m,
            year_of_loss=target.year,
            historical_storms=storms_chrono,
        )

        return CaseBenchmarkResult(
            target=target,
            metrics=metrics,
            post_depositional_site=post_site,
            seafloor_lats=seafloor_lats,
            seafloor_lons=seafloor_lons,
        )

    def run_all(self, rng_seed: int = 42) -> Dict[str, CaseBenchmarkResult]:
        """Execute all 4 curated historical benchmarks."""
        results: Dict[str, CaseBenchmarkResult] = {}
        for key, target in HISTORICAL_BENCHMARKS.items():
            results[key] = self.run_case(target, rng_seed=rng_seed)
        return results

    def generate_suite_report(self, results: Dict[str, CaseBenchmarkResult]) -> Dict[str, Any]:
        """Generate high-level aggregate benchmark validation report."""
        sar_values = [r.metrics.search_area_reduction_pct for r in results.values()]
        in_95_count = sum(1 for r in results.values() if r.metrics.contained_in_95_pct)
        in_80_count = sum(1 for r in results.values() if r.metrics.contained_in_80_pct)
        in_50_count = sum(1 for r in results.values() if r.metrics.contained_in_50_pct)
        errors_km = [r.metrics.centroid_error_km for r in results.values()]

        return {
            "total_benchmark_cases": len(results),
            "containment_rate_95_pct": round((in_95_count / len(results)) * 100.0, 1),
            "containment_rate_80_pct": round((in_80_count / len(results)) * 100.0, 1),
            "containment_rate_50_pct": round((in_50_count / len(results)) * 100.0, 1),
            "mean_search_area_reduction_pct": round(float(np.mean(sar_values)), 1),
            "min_search_area_reduction_pct": round(float(np.min(sar_values)), 1),
            "mean_centroid_error_km": round(float(np.mean(errors_km)), 2),
            "cases": {k: v.to_dict() for k, v in results.items()},
        }

    def to_geojson(self, results: Dict[str, CaseBenchmarkResult]) -> Dict[str, Any]:
        """Export all benchmark cases and simulated distributions as a combined GeoJSON FeatureCollection."""
        features: List[Dict[str, Any]] = []

        for key, res in results.items():
            t = res.target
            m = res.metrics

            # 1. Ground Truth Wreck Point
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(t.ground_truth_lon, 6), round(t.ground_truth_lat, 6)],
                },
                "properties": {
                    "case_key": key,
                    "wreck_name": t.name,
                    "year": t.year,
                    "feature_type": "archaeological_ground_truth_point",
                    "water_depth_m": t.water_depth_m,
                    "description": t.description,
                },
            })

            # 2. Predicted MAP Centroid Point
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(m.map_centroid_lon, 6), round(m.map_centroid_lat, 6)],
                },
                "properties": {
                    "case_key": key,
                    "wreck_name": t.name,
                    "feature_type": "predicted_map_centroid",
                    "centroid_error_km": m.centroid_error_km,
                    "search_area_reduction_pct": m.search_area_reduction_pct,
                },
            })

            # 3. Post-depositional site formation features if available
            if res.post_depositional_site is not None:
                post_gj = res.post_depositional_site.to_geojson()
                for f in post_gj["features"]:
                    f["properties"]["case_key"] = key
                    features.append(f)

        return {"type": "FeatureCollection", "features": features}
