"""Unit and integration tests for post-depositional site formation and storm dispersal."""

import json
import math
import pytest

from src.models.post_depositional import (
    PostDepositionalDispersalModel,
    PostDepositionalSiteState,
    compute_bottom_orbital_velocity,
    solve_wave_wavenumber,
)
from src.modules.extreme_event import TropicalCyclone


def test_wave_wavenumber_solver():
    """Verify Newton-Raphson wavenumber satisfies linear dispersion relation."""
    for depth in [5.0, 20.0, 80.0, 500.0]:
        tp = 8.0
        k = solve_wave_wavenumber(tp, depth)
        omega = 2.0 * math.pi / tp
        lhs = omega**2
        rhs = 9.80665 * k * math.tanh(k * depth)
        assert abs(lhs - rhs) < 1e-4

    with pytest.raises(ValueError, match="water_depth_m must be strictly positive"):
        solve_wave_wavenumber(8.0, 0.0)


def test_bottom_orbital_velocity_depth_attenuation():
    """Verify wave orbital motion reaches bottom in shallow water but decays to zero in deep water."""
    hs = 7.0  # 7m storm waves
    tp = 9.0

    # Shallow shelf (15m depth, e.g. Marquesas Keys reef)
    ub_shallow = compute_bottom_orbital_velocity(hs, tp, water_depth_m=15.0)
    assert ub_shallow > 1.0  # High bottom velocity (> 1 m/s), sufficient to roll silver coins

    # Intermediate depth (50m depth)
    ub_mid = compute_bottom_orbital_velocity(hs, tp, water_depth_m=50.0)
    assert 0.0 < ub_mid < ub_shallow

    # Deep abyssal plain (1500m depth, e.g. SS El Faro)
    ub_deep = compute_bottom_orbital_velocity(hs, tp, water_depth_m=1500.0)
    assert ub_deep == 0.0  # No surface wave disturbance on deep seafloor


def test_post_depositional_storm_replay_shallow_vs_deep():
    """Verify multi-century storm chronology mobilizes artifacts in shallow water while deep wrecks remain static."""
    model = PostDepositionalDispersalModel(
        artifact_drag_factor=0.015,
        coin_mobilization_threshold_ms=0.80,
    )

    storms = [
        # 1622 secondary October hurricane passing 30 km from Marquesas
        TropicalCyclone(
            storm_name="October_1622_Hurricane",
            year=1622,
            eye_lat=24.3,
            eye_lon=-82.1,
            v_max_m_s=55.0,
            p_central_hpa=942.0,
            r_max_km=40.0,
        ),
        # 1715 Florida hurricane
        TropicalCyclone(
            storm_name="Hurricane_1715",
            year=1715,
            eye_lat=25.0,
            eye_lon=-81.5,
            v_max_m_s=48.0,
            p_central_hpa=950.0,
            r_max_km=45.0,
        ),
    ]

    touchdown_lat = 24.50
    touchdown_lon = -82.20

    # 1. Shallow water case (17m, Atocha site)
    site_shallow = model.simulate_storm_chronology(
        initial_touchdown_lat=touchdown_lat,
        initial_touchdown_lon=touchdown_lon,
        water_depth_m=17.0,
        year_of_loss=1622,
        historical_storms=storms,
    )

    # Initial touchdown coordinates must be preserved
    assert site_shallow.touchdown_lat == touchdown_lat
    assert site_shallow.touchdown_lon == touchdown_lon
    # Mobile artifacts should have migrated
    assert site_shallow.cumulative_net_displacement_m > 50.0
    assert site_shallow.ballast_scour_depth_m > 0.0
    assert site_shallow.scatter_centroid_lat != touchdown_lat

    summary = site_shallow.summary()
    assert summary["mobilizing_storm_events"] > 0
    assert summary["net_artifact_migration_m"] > 50.0

    # 2. Deep water case (1000m, deep foundering)
    site_deep = model.simulate_storm_chronology(
        initial_touchdown_lat=touchdown_lat,
        initial_touchdown_lon=touchdown_lon,
        water_depth_m=1000.0,
        year_of_loss=1622,
        historical_storms=storms,
    )
    # In 1000m depth, zero bottom disturbance: displacement is exactly zero
    assert site_deep.cumulative_net_displacement_m == 0.0
    assert site_deep.scatter_centroid_lat == touchdown_lat
    assert site_deep.ballast_scour_depth_m == 0.0


def test_post_depositional_geojson_dual_layer_export():
    """Verify GeoJSON exports both initial touchdown site and migrated secondary scatter trail."""
    model = PostDepositionalDispersalModel()
    storms = [
        TropicalCyclone(
            storm_name="Test_Gale",
            year=1744,
            eye_lat=24.4,
            eye_lon=-81.4,
            v_max_m_s=50.0,
            p_central_hpa=945.0,
            r_max_km=35.0,
        )
    ]

    site = model.simulate_storm_chronology(
        initial_touchdown_lat=24.55,
        initial_touchdown_lon=-81.40,
        water_depth_m=12.0,
        year_of_loss=1744,
        historical_storms=storms,
    )

    geojson = site.to_geojson()
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) >= 4

    feature_types = {f["properties"]["feature_type"] for f in geojson["features"]}
    # Must retain BOTH initial touchdown site and migrated scatter centroid
    assert "initial_touchdown_site" in feature_types
    assert "ballast_mound_95_percent_contour" in feature_types
    assert "migrated_artifact_scatter_centroid" in feature_types
    assert "secondary_scatter_95_percent_contour" in feature_types

    # Formatted string validation
    json_str = site.to_geojson_str()
    parsed = json.loads(json_str)
    assert parsed["type"] == "FeatureCollection"
