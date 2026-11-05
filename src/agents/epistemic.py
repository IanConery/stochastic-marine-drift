"""Epistemic uncertainty prior calibration mapping archival credibility to spatial covariance matrices."""

from typing import Optional
from src.agents.judge import evaluate_extraction_quality
from src.agents.schemas import EpistemicPrior, VesselExtractionRecord

# Tier-to-uncertainty calibration table
# (1-sigma standard deviations in nautical miles and hours)
TIER_CALIBRATION_TABLE = {
    1: {
        "provenance_type": "eyewitness_captain_log",
        "spatial_std_nmi": 2.0,
        "temporal_std_hours": 1.0,
    },
    2: {
        "provenance_type": "escort_convoy_sighting",
        "spatial_std_nmi": 8.0,
        "temporal_std_hours": 6.0,
    },
    3: {
        "provenance_type": "colonial_inquest_rumor",
        "spatial_std_nmi": 30.0,
        "temporal_std_hours": 48.0,
    },
}


def calibrate_epistemic_prior(
    credibility_tier: int,
    custom_spatial_std_nmi: Optional[float] = None,
    custom_temporal_std_hours: Optional[float] = None,
) -> EpistemicPrior:
    """Instantiate a calibrated EpistemicPrior from an assigned credibility tier (1, 2, or 3).

    Args:
        credibility_tier: 1 (eyewitness), 2 (convoy sighting), 3 (retrospective rumor).
        custom_spatial_std_nmi: Optional override for spatial 1-sigma radius (nmi).
        custom_temporal_std_hours: Optional override for temporal 1-sigma uncertainty (hrs).

    Returns:
        EpistemicPrior instance with initialized covariance matrix.
    """
    tier = 2 if credibility_tier not in (1, 2, 3) else credibility_tier
    config = TIER_CALIBRATION_TABLE[tier]

    spatial_std = custom_spatial_std_nmi if custom_spatial_std_nmi is not None else config["spatial_std_nmi"]
    temporal_std = custom_temporal_std_hours if custom_temporal_std_hours is not None else config["temporal_std_hours"]

    return EpistemicPrior(
        source_credibility_tier=tier,  # type: ignore
        provenance_type=config["provenance_type"],
        spatial_prior_std_nmi=spatial_std,
        temporal_prior_std_hours=temporal_std,
    )


def derive_prior_from_text(
    raw_text: str,
    record: Optional[VesselExtractionRecord] = None,
    use_remote_judge: bool = True,
) -> EpistemicPrior:
    """Derive an epistemic prior directly from text using Kev/Jev decision evaluation.

    Args:
        raw_text: Source historical text.
        record: Optional VesselExtractionRecord.
        use_remote_judge: Whether to query remote Kev/Jev model.

    Returns:
        Calibrated EpistemicPrior.
    """
    dummy_record = record or VesselExtractionRecord(
        vessel_name="Evaluation Target", loss_narrative=raw_text
    )
    judge_eval = evaluate_extraction_quality(raw_text, dummy_record, use_remote_judge=use_remote_judge)

    return calibrate_epistemic_prior(credibility_tier=judge_eval.assigned_tier)
