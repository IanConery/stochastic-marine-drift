"""Unit and integration tests for discrete reverse-time Lagrangian backtracker."""

import json
import pytest

from src.models.backtrack import (
    BacktrackParticle,
    BacktrackSimulationResult,
    IntersectionResult,
    ReverseLagrangianBacktracker,
)
from src.models.forward_drift import (
    EnvironmentalConditions,
    ForwardDriftSimulator,
)
from src.profiles.presets import BERMUDA_SLOOP_1710, SPANISH_GALLEON_1620
from src.utils.geo import knots_to_m_s


def test_backtracker_initialization():
    """Verify default initialization parameters and speed threshold conversion."""
    bt = ReverseLagrangianBacktracker(
        profile=SPANISH_GALLEON_1620,
        max_velocity_knots=6.0,
        dt_seconds=300.0,
        num_particles=20,
        rng_seed=123,
    )
    assert bt.profile.class_name == "spanish_galleon"
    assert bt.num_particles == 20
    assert bt.dt_seconds == 300.0
    expected_ms = knots_to_m_s(6.0)
    assert abs(bt.max_velocity_ms - expected_ms) < 1e-4


def test_invalid_parameters_raise_value_error():
    """Verify negative depth or negative hours raise ValueError."""
    bt = ReverseLagrangianBacktracker()
    env = EnvironmentalConditions()

    with pytest.raises(ValueError, match="seafloor_depth_m must be strictly positive"):
        bt.run(seafloor_lat=24.5, seafloor_lon=-82.2, seafloor_depth_m=-100.0, total_hours=5.0, env=env)

    with pytest.raises(ValueError, match="total_hours must be strictly positive"):
        bt.run(seafloor_lat=24.5, seafloor_lon=-82.2, seafloor_depth_m=500.0, total_hours=0.0, env=env)


def test_vertical_ascent_phase():
    """Verify backtracker calculates Surface Point of Sink (SPS) displaced against deep currents."""
    bt = ReverseLagrangianBacktracker(
        profile=SPANISH_GALLEON_1620,
        num_particles=10,
        rng_seed=42,
    )
    # Deep currents flow East (u > 0)
    env = EnvironmentalConditions(
        deep_current_u_m_s=0.20,
        deep_current_v_m_s=0.0,
        u10_wind_m_s=0.0,
        v10_wind_m_s=0.0,
        u_ocean_current_m_s=0.0,
        v_ocean_current_m_s=0.0,
        wave_height_hs_m=0.1,
    )

    seafloor_lat = 24.500
    seafloor_lon = -82.200
    seafloor_depth = 600.0  # 600 m depth

    # If forward descent drifted East, reverse ascent backtracks West (lon < seafloor_lon)
    res = bt.run(
        seafloor_lat=seafloor_lat,
        seafloor_lon=seafloor_lon,
        seafloor_depth_m=seafloor_depth,
        total_hours=1.0,
        env=env,
        seafloor_uncertainty_m=0.0,  # Zero perturbation to verify exact directional shift
        glide_heading_deg=0.0,
    )

    assert len(res.particles) == 10
    mean_sps_lat, mean_sps_lon = res.mean_sps_coords
    # Surface Point of Sink must be West of seafloor touchdown because deep current was Eastward
    assert mean_sps_lon < seafloor_lon
    assert abs(mean_sps_lat - seafloor_lat) < 0.05


def test_bayesian_speed_bounding():
    """Verify excessive velocity forces are clipped to max_velocity_knots."""
    # Set extreme wind that would otherwise generate > 10 knot drift
    extreme_env = EnvironmentalConditions(
        u10_wind_m_s=45.0,  # Extreme hurricane force wind
        v10_wind_m_s=30.0,
        u_ocean_current_m_s=2.5,
        v_ocean_current_m_s=1.5,
        wave_height_hs_m=8.0,
    )

    bt = ReverseLagrangianBacktracker(
        profile=SPANISH_GALLEON_1620,
        max_velocity_knots=5.0,  # Strict 5-knot bound (~2.57 m/s)
        num_particles=5,
        dt_seconds=300.0,
        rng_seed=999,
    )

    res = bt.run(
        seafloor_lat=24.5,
        seafloor_lon=-82.2,
        seafloor_depth_m=400.0,
        total_hours=2.0,
        env=extreme_env,
    )

    # Assert that speed clipping occurred during reverse integration
    summary = res.summary()
    assert summary["total_speed_clipped_steps"] > 0
    assert summary["max_velocity_knots"] == 5.0


def test_geojson_export():
    """Verify GeoJSON export includes SPS points, origin estimates, and LineString tracks."""
    bt = ReverseLagrangianBacktracker(
        profile=BERMUDA_SLOOP_1710,
        num_particles=5,
        dt_seconds=600.0,
        rng_seed=42,
    )
    env = EnvironmentalConditions()

    res = bt.run(
        seafloor_lat=25.0,
        seafloor_lon=-80.0,
        seafloor_depth_m=300.0,
        total_hours=1.5,
        env=env,
    )

    geojson_dict = res.to_geojson()
    assert geojson_dict["type"] == "FeatureCollection"
    assert len(geojson_dict["features"]) >= 10  # Points + LineStrings

    json_str = res.to_geojson_str()
    parsed = json.loads(json_str)
    assert parsed["type"] == "FeatureCollection"

    feature_types = {f["properties"].get("feature_type") for f in parsed["features"]}
    assert "surface_point_of_sink" in feature_types
    assert "backtracked_origin_estimate" in feature_types
    assert "reverse_trajectory_track" in feature_types


def test_bidirectional_bayesian_closure_intersection():
    """Verify bi-directional coupling between forward drift and backtracker reduces search area."""
    # 1. Forward run
    fwd_sim = ForwardDriftSimulator(
        profile=SPANISH_GALLEON_1620,
        num_particles=25,
        dt_seconds=300.0,
        enable_crew_behavior=False,
        rng_seed=42,
    )
    env = EnvironmentalConditions(
        u10_wind_m_s=15.0,
        v10_wind_m_s=5.0,
        water_depth_m=200.0,
    )

    start_lat = 24.50
    start_lon = -82.20
    fwd_res = fwd_sim.run(
        lkp_lat=start_lat,
        lkp_lon=start_lon,
        total_hours=3.0,
        env=env,
    )

    # 2. Pick mean final location as input to backtracker
    import numpy as np
    final_lat = float(np.mean([p.lat for p in fwd_res.particles]))
    final_lon = float(np.mean([p.lon for p in fwd_res.particles]))

    # 3. Backtrack run
    bwd_sim = ReverseLagrangianBacktracker(
        profile=SPANISH_GALLEON_1620,
        num_particles=25,
        dt_seconds=300.0,
        rng_seed=42,
    )
    bwd_res = bwd_sim.run(
        seafloor_lat=final_lat,
        seafloor_lon=final_lon,
        seafloor_depth_m=200.0,
        total_hours=3.0,
        env=env,
    )

    # 4. Compute intersection
    intersection = bwd_res.compute_intersection(fwd_res)

    assert isinstance(intersection, IntersectionResult)
    assert intersection.forward_area_km2 > 0.0
    assert intersection.intersection_area_km2 > 0.0
    assert 0.0 <= intersection.search_area_reduction_pct <= 99.0
    # Search area reduction should be substantial (typically > 50%)
    assert intersection.search_area_reduction_pct > 30.0

    inter_dict = intersection.to_dict()
    assert "search_area_reduction_pct" in inter_dict
    assert "map_centroid_lat" in inter_dict
