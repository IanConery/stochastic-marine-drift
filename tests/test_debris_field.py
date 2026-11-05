"""Unit tests for bivariate Gaussian debris scatter footprints and GeoJSON export."""

import json
import math
import pytest

from src.models.debris_field import (
    DebrisField,
    estimate_debris_field,
    generate_confidence_ellipse_points,
    meters_to_lat_lon_offsets,
)
from src.profiles.presets import SPANISH_GALLEON_1620, MODERN_CONTAINER_2015


def test_meters_to_lat_lon_offsets():
    # 111,139 meters North should correspond to ~1.0 degree latitude
    d_lat, d_lon = meters_to_lat_lon_offsets(0.0, 111139.0, origin_lat=0.0)
    assert math.isclose(d_lat, 1.0, rel_tol=1e-3)
    assert d_lon == 0.0

    # East offset at equator
    d_lat_eq, d_lon_eq = meters_to_lat_lon_offsets(111139.0, 0.0, origin_lat=0.0)
    assert d_lat_eq == 0.0
    assert math.isclose(d_lon_eq, 1.0, rel_tol=1e-3)


def test_confidence_ellipse_closed_and_scaling():
    # Verify points close the polygon
    pts = generate_confidence_ellipse_points(
        center_x_m=0.0,
        center_y_m=0.0,
        semi_major_m=50.0,
        semi_minor_m=20.0,
        orientation_deg=45.0,
        confidence_level=0.95,
        num_points=24,
    )
    assert pts[0] == pts[-1]
    assert len(pts) == 25

    # Check scaling of area
    field = DebrisField(
        touchdown_lat=24.5,
        touchdown_lon=-81.5,
        ballast_semi_major_m=25.0,
        ballast_semi_minor_m=12.0,
        ballast_heading_deg=0.0,
        scatter_center_lat=24.51,
        scatter_center_lon=-81.51,
        scatter_semi_major_m=60.0,
        scatter_semi_minor_m=25.0,
        current_heading_deg=45.0,
    )
    area_50 = field.area_sq_meters("ballast", 0.50)
    area_80 = field.area_sq_meters("ballast", 0.80)
    area_95 = field.area_sq_meters("ballast", 0.95)

    assert area_95 > area_80 > area_50 > 0.0


def test_estimate_debris_field_downstream_displacement():
    touchdown_lat, touchdown_lon = 24.50, -81.50
    # Deep current Eastward (u=0.4 m/s, v=0)
    field = estimate_debris_field(
        touchdown_lat=touchdown_lat,
        touchdown_lon=touchdown_lon,
        deep_current_u_m_s=0.4,
        deep_current_v_m_s=0.0,
        descent_duration_s=1000.0,
        profile=SPANISH_GALLEON_1620,
    )

    # Ballast mound center is at touchdown
    assert field.touchdown_lat == touchdown_lat
    assert field.touchdown_lon == touchdown_lon

    # Scatter center should be shifted Eastward (scatter_lon > touchdown_lon)
    assert field.scatter_center_lon > touchdown_lon
    assert math.isclose(field.scatter_center_lat, touchdown_lat, abs_tol=1e-4)
    assert field.current_heading_deg == 90.0


def test_debris_field_geojson_schema():
    field = estimate_debris_field(
        touchdown_lat=24.50,
        touchdown_lon=-81.50,
        deep_current_u_m_s=0.2,
        deep_current_v_m_s=0.2,
        descent_duration_s=800.0,
        profile=MODERN_CONTAINER_2015,
        metadata={"vessel_name": "Test Wreck"},
    )

    geojson = field.to_geojson()
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 3

    # Feature 0: Touchdown point
    feat_pt = geojson["features"][0]
    assert feat_pt["geometry"]["type"] == "Point"
    assert feat_pt["properties"]["vessel_name"] == "Test Wreck"

    # Feature 1: Ballast 95% Polygon
    feat_ballast = geojson["features"][1]
    assert feat_ballast["geometry"]["type"] == "Polygon"
    poly_coords = feat_ballast["geometry"]["coordinates"][0]
    # Verify polygon closes
    assert poly_coords[0] == poly_coords[-1]

    # Verify JSON serialization
    json_str = field.to_geojson_str()
    parsed = json.loads(json_str)
    assert parsed["type"] == "FeatureCollection"
