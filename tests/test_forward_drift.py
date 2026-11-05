"""Unit and integration tests for Phase 5 forward Monte Carlo drift engine and crew state machine."""

import json
import math
import numpy as np
import pytest

from src.agents.schemas import EpistemicPrior
from src.models.forward_drift import (
    DriftParticle,
    EnvironmentalConditions,
    ForwardDriftSimulator,
    ParticleState,
)
from src.modules.collision import BathymetricGrid
from src.profiles.presets import BERMUDA_SLOOP_1710, SPANISH_GALLEON_1620


# =========================================================================
# 1. Initialization & Seeding Tests
# =========================================================================

def test_simulator_initialization_with_epistemic_prior():
    prior = EpistemicPrior(
        source_credibility_tier=1,
        provenance_type="eyewitness_captain_log",
        spatial_prior_std_nmi=2.0,
    )
    sim = ForwardDriftSimulator(
        profile=SPANISH_GALLEON_1620,
        prior=prior,
        num_particles=50,
        rng_seed=42,
    )
    particles = sim._initialize_particles(lkp_lat=24.5, lkp_lon=-81.5)

    assert len(particles) == 50
    # Mean position of particles should center on LKP within 2 nmi (~0.03 deg)
    mean_lat = float(np.mean([p.lat for p in particles]))
    mean_lon = float(np.mean([p.lon for p in particles]))
    assert math.isclose(mean_lat, 24.5, abs_tol=0.03)
    assert math.isclose(mean_lon, -81.5, abs_tol=0.03)


# =========================================================================
# 2. Physics & Directional Drift Tests
# =========================================================================

def test_eastward_wind_drives_eastward_drift():
    # Strong Eastward wind (u=15 m/s, v=0) and calm currents
    env = EnvironmentalConditions(
        u10_wind_m_s=15.0,
        v10_wind_m_s=0.0,
        u_ocean_current_m_s=0.0,
        v_ocean_current_m_s=0.0,
        wave_height_hs_m=1.0,
        wave_period_tp_s=6.0,
        wave_direction_deg=90.0,  # Eastward wave
        water_depth_m=1000.0,      # Deep water (no grounding)
    )

    sim = ForwardDriftSimulator(
        profile=SPANISH_GALLEON_1620,
        num_particles=20,
        dt_seconds=600.0,
        enable_crew_behavior=False,
        rng_seed=123,
    )
    result = sim.run(lkp_lat=24.0, lkp_lon=-80.0, total_hours=6.0, env=env)

    # After 6 hours of Eastward wind, all particles should have drifted East (lon > -80.0)
    for p in result.particles:
        assert p.lon > -80.0
        # Latitude change should be minor compared to dominant Eastward displacement
        assert abs(p.lat - 24.0) < 0.2


# =========================================================================
# 3. Grounding on Reef Tests
# =========================================================================

def test_shallow_bathymetry_triggers_grounding():
    # Shallow charted water (3.5m) with a galleon draft of 4.8m -> must ground immediately
    env = EnvironmentalConditions(
        water_depth_m=3.5,
        wave_height_hs_m=1.0,
        u10_wind_m_s=10.0,
        v10_wind_m_s=0.0,
    )

    sim = ForwardDriftSimulator(
        profile=SPANISH_GALLEON_1620,
        num_particles=15,
        dt_seconds=300.0,
        rng_seed=99,
    )
    result = sim.run(lkp_lat=24.5, lkp_lon=-81.5, total_hours=2.0, env=env)

    # 100% of particles should be grounded
    assert result.summary()["grounded_percentage"] == 100.0
    for p in result.particles:
        assert p.state == ParticleState.GROUNDED_ON_REEF
        assert p.is_terminal is True
        assert p.terminal_info["cause"] == "shallow_shoal"


# =========================================================================
# 4. Crew State Machine & Foundering Tests
# =========================================================================

def test_crew_state_transitions_and_cargo_jettison():
    # Severe Category 3 hurricane forcing: 35 m/s winds, 8m waves
    env = EnvironmentalConditions(
        u10_wind_m_s=35.0,
        v10_wind_m_s=10.0,
        wave_height_hs_m=8.0,
        wave_period_tp_s=10.0,
        water_depth_m=1200.0,
    )

    sim = ForwardDriftSimulator(
        profile=SPANISH_GALLEON_1620,
        num_particles=25,
        dt_seconds=300.0,
        enable_crew_behavior=True,
        rng_seed=42,
    )
    result = sim.run(lkp_lat=23.0, lkp_lon=-82.0, total_hours=12.0, env=env)

    # Under 12 hours of extreme storm conditions, particles should experience
    # severe hull fatigue and founder
    foundered = result.foundered_particles
    assert len(foundered) > 0

    for p in foundered:
        assert p.state == ParticleState.FOUNDERED
        assert p.seafloor_lat is not None
        assert p.seafloor_lon is not None
        # Verify 3D descent generated a debris field
        assert p.debris_field is not None
        assert p.debris_field.area_sq_meters("ballast", 0.95) > 0.0


# =========================================================================
# 5. Output Schemas & Probability Density Grid Tests
# =========================================================================

def test_geojson_export_and_density_grid():
    env = EnvironmentalConditions(
        u10_wind_m_s=14.0,
        v10_wind_m_s=5.0,
        water_depth_m=2.0,  # Shallow water (2.0m) triggers reef grounding for 2.6m draft sloop
        wave_height_hs_m=2.0,
    )

    sim = ForwardDriftSimulator(
        profile=BERMUDA_SLOOP_1710,
        num_particles=10,
        dt_seconds=600.0,
        rng_seed=1,
    )
    result = sim.run(lkp_lat=24.5, lkp_lon=-81.0, total_hours=3.0, env=env)

    # 1. GeoJSON serialization
    geojson_str = result.to_geojson_str()
    parsed = json.loads(geojson_str)
    assert parsed["type"] == "FeatureCollection"
    assert len(parsed["features"]) >= 10

    # 2. Probability density grid
    lat_bins, lon_bins, density = result.compute_probability_density_grid(grid_resolution_deg=0.05)
    assert len(lat_bins) > 1
    assert len(lon_bins) > 1
    # Total probability mass across density grid should sum to ~1.0
    assert math.isclose(float(np.sum(density)), 1.0, rel_tol=1e-4)
