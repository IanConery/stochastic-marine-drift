"""Unit and integration tests for Phase 10 retraining pipeline and evaluation metrics."""

import os
from pathlib import Path
import pytest
import pandas as pd

from src.models.retrain_pipeline import (
    extract_features_from_corpus,
    map_archetype_to_profile,
    retrain_models,
    run_backtrack_for_training_wrecks,
    RetrainingMetrics,
)
from src.profiles.presets import SPANISH_GALLEON_1620, BERMUDA_SLOOP_1710


def test_map_archetype_to_profile():
    """Verify macro-archetype strings correctly resolve to physics vessel profiles."""
    p_3m = map_archetype_to_profile("3_mast_square")
    assert p_3m.class_name == SPANISH_GALLEON_1620.class_name

    p_1m = map_archetype_to_profile("1_mast_fore_aft")
    assert p_1m.class_name == BERMUDA_SLOOP_1710.class_name

    p_unknown = map_archetype_to_profile("unknown_fantasy_rig")
    assert p_unknown.class_name == SPANISH_GALLEON_1620.class_name


def test_extract_features_from_corpus():
    """Verify feature extraction from extracted historical corpus preserves schemas and partitions."""
    df = extract_features_from_corpus(
        jsonl_path="data/raw/wikipedia_maritime_extracts.jsonl",
        geojson_path="data/processed/expanded_shipwreck_catalog.geojson",
        rng_seed=42,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 300
    assert "partition" in df.columns
    assert "event_observed" in df.columns
    assert "duration_hours" in df.columns
    assert "hs_mean_m" in df.columns

    train_count = (df["partition"] == "train").sum()
    test_count = (df["partition"] == "test").sum()
    assert train_count > 0
    assert test_count > 0
    assert 0.65 <= train_count / len(df) <= 0.85


def test_run_backtrack_for_training_wrecks():
    """Verify Lagrangian backtracker generates valid SPS coordinates on historical casualties."""
    sample_df = pd.DataFrame([
        {
            "vessel_name": "Test Atocha",
            "macro_archetype": "3_mast_square",
            "partition": "train",
            "duration_hours": 4.0,
            "event_observed": 1,
            "hs_mean_m": 4.5,
            "tp_mean_s": 8.0,
            "wave_steepness": 0.045,
            "wind_speed_mean_ms": 25.0,
            "vessel_tonnage": 550.0,
            "freeboard_m": 1.5,
            "cumulative_stress": 45.0,
            "water_depth_m": 17.0,
            "lat": 24.5292,
            "lon": -82.2525,
        }
    ])
    results = run_backtrack_for_training_wrecks(sample_df, max_cases=1)
    assert len(results) == 1
    res = results[0]
    assert res["vessel_name"] == "Test Atocha"
    assert "sps_lat" in res and res["sps_lat"] != 0.0
    assert "sps_lon" in res and res["sps_lon"] != 0.0
    assert res["valid_particles"] > 0


def test_retrain_models_evaluation_metrics(tmp_path: Path):
    """Verify that retrain_models runs end-to-end, writing models and returning valid evaluation metrics."""
    foundering_out = str(tmp_path / "foundering_model.pkl")
    gnn_out = str(tmp_path / "convoy_gnn.pt")
    metrics_out = str(tmp_path / "metrics.json")

    metrics = retrain_models(
        jsonl_path="data/raw/wikipedia_maritime_extracts.jsonl",
        catalog_path="data/processed/expanded_shipwreck_catalog.geojson",
        foundering_out=foundering_out,
        gnn_out=gnn_out,
        metrics_out=metrics_out,
    )

    assert isinstance(metrics, RetrainingMetrics)
    assert metrics.train_count > 100
    assert metrics.test_count > 30
    assert metrics.harrell_c_index >= 0.70
    assert 0.0 <= metrics.brier_score <= 1.0
    assert metrics.mean_sar_reduction_pct >= 85.0
    assert os.path.exists(foundering_out)
    assert os.path.exists(gnn_out)
    assert os.path.exists(metrics_out)
