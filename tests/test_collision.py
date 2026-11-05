"""Unit tests for bathymetric collision, wave trough plunge, and reef grounding filter."""

import numpy as np
import pytest

from src.modules.collision import (
    BathymetricGrid,
    check_particle_grounding,
    evaluate_grounding_condition,
)


def test_evaluate_grounding_calm_deep_water():
    is_grounded, eff_depth, cause = evaluate_grounding_condition(
        water_depth_m=25.0, vessel_draft_m=5.0, tide_height_m=0.0, wave_height_m=0.0
    )
    assert not is_grounded
    assert eff_depth == 25.0
    assert cause == "none"


def test_evaluate_grounding_shallow_shoal():
    # Keel draft 4.8m in 3.5m charted water
    is_grounded, eff_depth, cause = evaluate_grounding_condition(
        water_depth_m=3.5, vessel_draft_m=4.8, tide_height_m=0.0, wave_height_m=0.0
    )
    assert is_grounded
    assert eff_depth == 3.5
    assert cause == "shallow_shoal"


def test_evaluate_grounding_wave_trough_plunge():
    # Charted depth 6.0m clears 5.0m keel draft in calm seas (underkeel clearance 1.0m)
    # But a 4.0m hurricane wave creates a 2.0m trough plunge: effective depth = 4.0m <= 5.0m
    is_grounded, eff_depth, cause = evaluate_grounding_condition(
        water_depth_m=6.0, vessel_draft_m=5.0, tide_height_m=0.0, wave_height_m=4.0
    )
    assert is_grounded
    assert eff_depth == 4.0
    assert cause == "reef_collision_wave_trough"


def test_evaluate_grounding_low_tide():
    # Charted depth 5.5m with draft 5.0m. Under spring low tide of -1.2m, effective depth is 4.3m
    is_grounded, eff_depth, cause = evaluate_grounding_condition(
        water_depth_m=5.5, vessel_draft_m=5.0, tide_height_m=-1.2, wave_height_m=0.0
    )
    assert is_grounded
    assert eff_depth == 4.3
    assert cause == "shallow_shoal"


def test_evaluate_grounding_landfall():
    # Depth <= 0 represents dry land or emergent reef crest
    is_grounded, eff_depth, cause = evaluate_grounding_condition(
        water_depth_m=-1.5, vessel_draft_m=3.0, tide_height_m=0.0, wave_height_m=0.0
    )
    assert is_grounded
    assert cause == "landfall"


def test_bathymetric_grid_bilinear_sampling():
    lats = np.array([24.0, 25.0])
    lons = np.array([-81.0, -80.0])
    # Corner depths: 10m (SW), 20m (SE), 30m (NW), 40m (NE)
    depths = np.array([[10.0, 20.0], [30.0, 40.0]])
    grid = BathymetricGrid(lats=lats, lons=lons, depth_matrix_m=depths)

    # Center point (24.5, -80.5) must be exact average of 4 corners = 25.0m
    mid_depth = grid.sample_depth_bilinear(24.5, -80.5)
    assert mid_depth == 25.0

    # Test SW corner
    assert grid.sample_depth_bilinear(24.0, -81.0) == 10.0


def test_check_particle_grounding_geojson():
    event = check_particle_grounding(
        lat=24.55,
        lon=-81.42,
        vessel_draft_m=4.8,
        bathymetry_source=3.0,  # 3m depth
        wave_height_m=2.0,
        timestamp_s=3600.0,
    )
    assert event.is_grounded
    assert event.cause == "shallow_shoal"

    geojson = event.to_geojson_feature()
    assert geojson["type"] == "Feature"
    assert geojson["geometry"]["coordinates"] == [-81.42, 24.55]
    assert geojson["properties"]["is_grounded"] is True
    assert geojson["properties"]["under_keel_clearance_m"] < 0
