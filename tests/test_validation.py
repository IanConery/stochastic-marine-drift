"""Integration tests for historical ground-truth benchmark suite."""

import json
import pytest

from src.models.validation import (
    HISTORICAL_BENCHMARKS,
    HistoricalBenchmarkRunner,
)


def test_benchmark_targets_registry():
    """Verify registry contains all 4 documented benchmark losses with valid archaeological coordinates."""
    expected_keys = ["atocha", "santa_margarita", "hms_looe", "el_faro"]
    for k in expected_keys:
        assert k in HISTORICAL_BENCHMARKS
        target = HISTORICAL_BENCHMARKS[k]
        assert target.year > 1500
        assert target.water_depth_m > 0
        assert target.naive_uncertainty_radius_nmi >= 15.0
        assert -90.0 <= target.ground_truth_lat <= 90.0
        assert -180.0 <= target.ground_truth_lon <= 180.0


def test_atocha_benchmark_case():
    """Verify Atocha 1622 benchmark containment within 95% contour and > 85% search area reduction."""
    runner = HistoricalBenchmarkRunner(num_particles=25, dt_seconds=300.0)
    target = HISTORICAL_BENCHMARKS["atocha"]

    res = runner.run_case(target, rng_seed=42)

    assert res.target.name == "Nuestra Senora de Atocha"
    metrics = res.metrics
    # 95% probability contour must contain ground truth
    assert metrics.contained_in_95_pct is True
    # Search Area Reduction must meet framework threshold (>= 85%)
    assert metrics.search_area_reduction_pct >= 85.0
    assert metrics.centroid_error_km < 35.0

    # Post-depositional secondary scatter must be generated
    assert res.post_depositional_site is not None
    assert res.post_depositional_site.cumulative_net_displacement_m > 0.0


def test_full_benchmark_suite_execution_and_report():
    """Verify execution of all 4 benchmarks achieves >= 85% mean SAR and 100% 95% containment rate."""
    runner = HistoricalBenchmarkRunner(num_particles=20, dt_seconds=300.0)
    results = runner.run_all(rng_seed=42)

    assert len(results) == 4
    report = runner.generate_suite_report(results)

    assert report["total_benchmark_cases"] == 4
    # All 4 cases must fall within the 95% simulated probability contour
    assert report["containment_rate_95_pct"] == 100.0
    # Mean Search Area Reduction must exceed 85%
    assert report["mean_search_area_reduction_pct"] >= 85.0

    # Combined GeoJSON FeatureCollection verification
    geojson = runner.to_geojson(results)
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) >= 10

    # Verify both ground truth and predicted centroids are present
    feature_types = {f["properties"].get("feature_type") for f in geojson["features"]}
    assert "archaeological_ground_truth_point" in feature_types
    assert "predicted_map_centroid" in feature_types
