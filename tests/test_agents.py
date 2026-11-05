"""Unit tests for Phase 4 archival extraction, Kev/Jev judge, and epistemic priors."""

import math
import numpy as np
import pytest

from src.agents.epistemic import (
    calibrate_epistemic_prior,
    derive_prior_from_text,
)
from src.agents.judge import (
    evaluate_extraction_quality,
    heuristic_judge_evaluation,
)
from src.agents.parser import (
    extract_vessel_record,
    heuristic_text_parser,
)
from src.agents.schemas import (
    EpistemicPrior,
    VesselExtractionRecord,
)
from src.profiles.vessel_class import MacroArchetype, RigType


# =========================================================================
# 1. Schema Tests
# =========================================================================

def test_vessel_extraction_record_to_profile():
    record = VesselExtractionRecord(
        vessel_name="HMS Looe",
        wreck_year=1744,
        mast_count=3,
        rig_type=RigType.SQUARE,
        is_modern=False,
        tonnage=850.0,
        cargo_mass_category="heavy",
        draft_meters=5.1,
    )
    profile = record.to_vessel_profile()

    assert profile.class_name == "hms_looe"
    assert not profile.is_modern
    assert profile.mast_count == 3
    assert profile.draft_meters == 5.1
    assert profile.macro_archetype == MacroArchetype.THREE_MAST_SQUARE


def test_epistemic_prior_covariance_and_sampling():
    prior = EpistemicPrior(
        source_credibility_tier=1,
        provenance_type="eyewitness_captain_log",
        spatial_prior_std_nmi=2.0,
    )
    # 2.0 nmi = 3704 meters; variance = 3704^2 = 13,719,616 m^2
    var_m2 = prior.spatial_covariance_matrix_m2[0][0]
    expected_var = (2.0 * 1852.0) ** 2
    assert math.isclose(var_m2, expected_var, rel_tol=1e-5)

    # Sample position around Havana (23.0 N, -82.0 W)
    rng = np.random.default_rng(42)
    sample_lat, sample_lon = prior.sample_initial_position(23.0, -82.0, rng=rng)
    assert 22.95 < sample_lat < 23.05
    assert -82.05 < sample_lon < -81.95


# =========================================================================
# 2. Parser Tests
# =========================================================================

def test_heuristic_parser_spanish_galleon():
    text = (
        "In the disastrous storm of September 1622, the galleon Nuestra Senora de Atocha "
        "carrying 550 tons of silver bars and gold specie was driven leeward onto the reefs "
        "near 24.55 N, 81.92 W."
    )
    record = heuristic_text_parser(text)

    assert "Atocha" in record.vessel_name or "galleon" in record.vessel_name.lower()
    assert record.wreck_year == 1622
    assert record.mast_count == 3
    assert record.rig_type == RigType.SQUARE
    assert not record.is_modern
    assert record.tonnage == 550.0
    assert record.cargo_mass_category == "precious_metals"
    assert record.lkp_lat == 24.55
    assert record.lkp_lon == -81.92


def test_heuristic_parser_bermuda_sloop():
    text = "A small privateer sloop of one mast and 65 tons was sighted in 1718."
    record = heuristic_text_parser(text)

    assert record.mast_count == 1
    assert record.rig_type == RigType.FORE_AND_AFT
    assert record.tonnage == 65.0
    assert not record.is_modern


def test_heuristic_parser_modern_container():
    text = "The modern steel container vessel was lost during severe swells in 2015."
    record = heuristic_text_parser(text)

    assert record.is_modern
    assert record.mast_count == 0
    assert record.rig_type == RigType.UNMASTED_POWER


# =========================================================================
# 3. Judge & Kev/Jev System-One Tests
# =========================================================================

def test_heuristic_judge_hallucination_detection():
    text = "A small coastal sloop of single mast was dismasted."
    # Extractor wrongly hallucinated mast_count=3
    hallucinated_record = VesselExtractionRecord(
        vessel_name="Sloop", mast_count=3, rig_type=RigType.SQUARE
    )
    evaluation = heuristic_judge_evaluation(text, hallucinated_record)

    assert evaluation.hallucination_detected is True
    assert evaluation.is_factual is False
    assert evaluation.confidence_score <= 0.50


def test_heuristic_judge_provenance_tiering():
    # Eyewitness captain journal -> Tier 1
    t1_text = "At 4 bells, we struck the reef and our rudder was carried away. I ordered the masts cut."
    t1_record = VesselExtractionRecord(vessel_name="HMS Looe", mast_count=3)
    eval_t1 = heuristic_judge_evaluation(t1_text, t1_record)
    assert eval_t1.assigned_tier == 1
    assert eval_t1.confidence_score >= 0.90

    # Retrospective colonial rumor -> Tier 3
    t3_text = "Some months later, rumors reached the Governor in Havana that the ship had foundered."
    t3_record = VesselExtractionRecord(vessel_name="Santa Margarita", mast_count=3)
    eval_t3 = heuristic_judge_evaluation(t3_text, t3_record)
    assert eval_t3.assigned_tier == 3


# =========================================================================
# 4. Epistemic Prior Calibration Tests
# =========================================================================

def test_calibrate_epistemic_prior_tiers():
    prior_t1 = calibrate_epistemic_prior(1)
    prior_t2 = calibrate_epistemic_prior(2)
    prior_t3 = calibrate_epistemic_prior(3)

    assert prior_t1.spatial_prior_std_nmi == 2.0
    assert prior_t2.spatial_prior_std_nmi == 8.0
    assert prior_t3.spatial_prior_std_nmi == 30.0

    assert prior_t1.temporal_prior_std_hours < prior_t2.temporal_prior_std_hours < prior_t3.temporal_prior_std_hours


def test_derive_prior_from_text_end_to_end():
    eyewitness_log = "Captain's Log: Our journal records that at midnight we took on water in the hold."
    prior = derive_prior_from_text(eyewitness_log, use_remote_judge=False)

    assert prior.source_credibility_tier == 1
    assert prior.provenance_type == "eyewitness_captain_log"
    assert prior.spatial_prior_std_nmi == 2.0
