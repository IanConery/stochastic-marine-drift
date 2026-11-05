"""Unit tests for search theory evaluation metrics (containment and SAR)."""

import pytest

from src.utils.metrics import (
    compute_centroid_error,
    compute_containment_status,
    compute_search_area_reduction,
)


def test_centroid_error_calculation():
    """Verify great-circle distance calculation between predicted centroid and ground truth."""
    # 1 degree of latitude is approximately 111.139 km
    err_km, err_nmi = compute_centroid_error(
        predicted_lat=24.0,
        predicted_lon=-82.0,
        true_lat=25.0,
        true_lon=-82.0,
    )
    assert abs(err_km - 111.19) < 1.0
    assert abs(err_nmi - 60.0) < 1.0


def test_containment_status_percentiles():
    """Verify ground truth is correctly identified within 50%, 80%, and 95% spatial radii."""
    # Symmetrically distributed particle cloud centered around (24.5, -82.0)
    sim_lats = [24.48, 24.50, 24.52, 24.49, 24.51, 24.50, 24.47, 24.53]
    sim_lons = [-82.02, -82.00, -81.98, -82.01, -81.99, -82.00, -82.03, -81.97]

    # Target right at center: should be contained in all 3 percentiles
    in_50, in_80, in_95, area_km2, _, _ = compute_containment_status(
        simulated_lats=sim_lats,
        simulated_lons=sim_lons,
        true_lat=24.50,
        true_lon=-82.00,
    )
    assert in_50 is True
    assert in_80 is True
    assert in_95 is True
    assert area_km2 > 0.0

    # Target far away: should not be contained
    in_50_far, in_80_far, in_95_far, _, _, _ = compute_containment_status(
        simulated_lats=sim_lats,
        simulated_lons=sim_lons,
        true_lat=26.00,
        true_lon=-80.00,
    )
    assert in_50_far is False
    assert in_80_far is False
    assert in_95_far is False


def test_search_area_reduction_percentage():
    """Verify Search Area Reduction exceeds 85% when model contour is significantly smaller than naive circle."""
    # Naive search area: 20 nautical mile radius circle (area ~ 4300 km^2)
    # Model 95% contour: ~100 km^2
    naive_radius_nmi = 20.0
    model_area_km2 = 100.0

    naive_area_km2, sar_pct = compute_search_area_reduction(
        naive_uncertainty_radius_nmi=naive_radius_nmi,
        model_area_95_km2=model_area_km2,
    )

    assert naive_area_km2 > 4000.0
    # SAR should exceed 95%
    assert sar_pct > 90.0
