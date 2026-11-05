"""Unit and integration tests for dynamic foundering survival classifier and partial pooling."""

import os
import tempfile
import pandas as pd
import pytest

from src.models.forward_drift import EnvironmentalConditions, ForwardDriftSimulator, ParticleState
from src.models.foundering import (
    BayesianPartialPooler,
    DynamicFounderingModel,
    SyntheticCohortGenerator,
    train_and_save_default_model,
)
from src.profiles.presets import BERMUDA_SLOOP_1710, SPANISH_GALLEON_1620


def test_synthetic_cohort_generation():
    """Verify synthetic historical cohort has valid schema, distributions, and archetype coverage."""
    df = SyntheticCohortGenerator.generate_cohort(num_cases=50, rng_seed=42)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50

    expected_cols = [
        "vessel_name",
        "vessel_subtype",
        "macro_archetype",
        "duration_hours",
        "event_observed",
        "hs_mean_m",
        "wave_steepness",
        "wind_speed_mean_ms",
        "vessel_tonnage",
        "freeboard_m",
        "cumulative_stress",
    ]
    for col in expected_cols:
        assert col in df.columns

    # Verify duration and wave values are positive
    assert (df["duration_hours"] > 0).all()
    assert (df["hs_mean_m"] > 0).all()
    assert (df["wave_steepness"] > 0).all()
    # Events must be binary 0 or 1
    assert set(df["event_observed"].unique()).issubset({0, 1})

    # Assert multiple macro-archetypes are represented
    archetypes = set(df["macro_archetype"].unique())
    assert len(archetypes) >= 3


def test_bayesian_partial_pooling_shrinkage():
    """Verify rare subtypes shrink toward macro-archetype mean, while unseen subtypes fall back safely."""
    df = SyntheticCohortGenerator.generate_cohort(num_cases=60, rng_seed=123)
    pooler = BayesianPartialPooler(shrinkage_kappa=3.0)
    pooler.fit(df)

    # All macro-archetypes present in data must have priors
    for arch in df["macro_archetype"].unique():
        assert str(arch) in pooler.archetype_priors
        assert pooler.archetype_priors[str(arch)] > 0.0

    # Rare / unseen subtype test
    unseen_mult = pooler.get_hazard_multiplier("unknown_ghost_ship", macro_archetype="3_mast_square")
    # Must match the archetype prior
    assert unseen_mult == pooler.archetype_priors["3_mast_square"]

    # Completely unknown subtype and archetype falls back to neutral 1.0
    neutral_mult = pooler.get_hazard_multiplier("phantom_ship", macro_archetype="nonexistent_rig")
    assert neutral_mult == 1.0


def test_dynamic_foundering_model_fit_and_monotonicity():
    """Verify survival model fitting and monotonic hazard response to severe sea states."""
    df = SyntheticCohortGenerator.generate_cohort(num_cases=70, rng_seed=42)
    model = DynamicFounderingModel(shrinkage_kappa=3.0)
    assert not model.is_fitted

    model.fit(df)
    assert model.is_fitted

    # Compare calm sea state vs extreme hurricane sea state
    calm_prob = model.predict_instantaneous_hazard(
        hs_m=1.0,
        tp_s=12.0,
        wind_speed_ms=5.0,
        freeboard_m=3.0,
        cumulative_stress=10.0,
        vessel_subtype="spanish_galleon",
        macro_archetype="3_mast_square",
        dt_seconds=600.0,
    )

    storm_prob = model.predict_instantaneous_hazard(
        hs_m=8.5,
        tp_s=6.0,  # Short steep cross-seas
        wind_speed_ms=35.0,
        freeboard_m=0.8,  # Heavily laden, low freeboard
        cumulative_stress=120.0,
        vessel_subtype="spanish_galleon",
        macro_archetype="3_mast_square",
        dt_seconds=600.0,
    )

    assert 0.0 <= calm_prob <= 1.0
    assert 0.0 <= storm_prob <= 1.0
    # Severe storm hazard must strictly exceed calm sea hazard
    assert storm_prob > calm_prob


def test_survival_curve_decreases_over_time():
    """Verify predicted survival probability S(t) decreases monotonically over elapsed duration."""
    df = SyntheticCohortGenerator.generate_cohort(num_cases=60, rng_seed=42)
    model = DynamicFounderingModel().fit(df)

    surv_2hr = model.predict_survival_probability(
        hs_m=5.0,
        tp_s=8.0,
        wind_speed_ms=18.0,
        freeboard_m=2.0,
        cumulative_stress=30.0,
        duration_hours=2.0,
        vessel_subtype="spanish_galleon",
        macro_archetype="3_mast_square",
    )

    surv_24hr = model.predict_survival_probability(
        hs_m=5.0,
        tp_s=8.0,
        wind_speed_ms=18.0,
        freeboard_m=2.0,
        cumulative_stress=30.0,
        duration_hours=24.0,
        vessel_subtype="spanish_galleon",
        macro_archetype="3_mast_square",
    )

    assert 0.0 <= surv_2hr <= 1.0
    assert 0.0 <= surv_24hr <= 1.0
    assert surv_24hr <= surv_2hr


def test_model_serialization():
    """Verify model saves to disk and reloads with identical predictions."""
    df = SyntheticCohortGenerator.generate_cohort(num_cases=50, rng_seed=99)
    model = DynamicFounderingModel().fit(df)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        model.save(tmp_path)
        assert os.path.exists(tmp_path)

        loaded_model = DynamicFounderingModel.load(tmp_path)
        assert loaded_model.is_fitted

        pred_orig = model.predict_instantaneous_hazard(
            hs_m=4.5,
            tp_s=7.0,
            wind_speed_ms=16.0,
            freeboard_m=1.8,
            cumulative_stress=40.0,
            vessel_subtype="bermuda_sloop",
            macro_archetype="one_mast_fore_aft",
        )
        pred_loaded = loaded_model.predict_instantaneous_hazard(
            hs_m=4.5,
            tp_s=7.0,
            wind_speed_ms=16.0,
            freeboard_m=1.8,
            cumulative_stress=40.0,
            vessel_subtype="bermuda_sloop",
            macro_archetype="one_mast_fore_aft",
        )

        assert abs(pred_orig - pred_loaded) < 1e-6
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_forward_drift_integration_with_survival_model():
    """Verify ForwardDriftSimulator dynamically founders particles using survival model."""
    df = SyntheticCohortGenerator.generate_cohort(num_cases=60, rng_seed=42)
    survival_model = DynamicFounderingModel().fit(df)

    sim = ForwardDriftSimulator(
        profile=SPANISH_GALLEON_1620,
        survival_model=survival_model,
        num_particles=20,
        dt_seconds=300.0,
        enable_crew_behavior=True,
        rng_seed=42,
    )

    # Severe hurricane conditions in deep water (no reef grounding)
    env = EnvironmentalConditions(
        u10_wind_m_s=35.0,
        v10_wind_m_s=25.0,
        wave_height_hs_m=9.0,
        wave_period_tp_s=6.5,  # High steepness
        water_depth_m=1200.0,
    )

    res = sim.run(
        lkp_lat=24.50,
        lkp_lon=-82.20,
        total_hours=4.0,
        env=env,
    )

    summary = res.summary()
    assert summary["total_particles"] == 20
    # Under extreme 9m waves and 43 m/s winds, particles should experience foundering
    assert summary["foundered_count"] > 0
    foundered = res.foundered_particles
    assert len(foundered) > 0
    # Foundered particles must have touchdown coordinates and debris field generated
    sample = foundered[0]
    assert sample.seafloor_lat is not None
    assert sample.seafloor_lon is not None
    assert sample.debris_field is not None
