"""Pydantic data schemas for archival text extraction, epistemic priors, and judge evaluations."""

import math
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field

from src.profiles.presets import profile_from_masts_and_rig
from src.profiles.vessel_class import RigType, VesselProfile

# Conversion constant: 1 nautical mile in meters
METERS_PER_NAUTICAL_MILE = 1852.0


class VesselExtractionRecord(BaseModel):
    """Structured parameters extracted from historical logbooks and wreck registries.

    Attributes:
        vessel_name: Name or designation of the ship.
        wreck_year: Year of casualty if identifiable.
        mast_count: Number of masts (0 for modern powered vessels, 1, 2, or 3+).
        rig_type: Sail plan orientation ('square', 'fore_and_aft', 'unmasted_power').
        is_modern: True for steel/powered vessels, False for historical wooden hulls.
        tonnage: Estimated displacement or burden in tons.
        cargo_mass_category: Cargo loading ('light', 'medium', 'heavy', 'precious_metals').
        buoyancy_factor: Ratio of reserve buoyancy (0.0 to 1.0).
        draft_meters: Keel draft depth in meters if mentioned.
        lkp_lat: Last known position latitude in degrees.
        lkp_lon: Last known position longitude in degrees.
        loss_narrative: Verbatim or summarized archival text describing the loss.
    """

    vessel_name: str = Field(description="Name or identifier of the ship")
    wreck_year: Optional[int] = Field(default=None, description="Year of loss")
    mast_count: int = Field(default=3, ge=0, description="Number of masts (0 for modern powered)")
    rig_type: RigType = Field(default=RigType.SQUARE, description="Sail plan orientation")
    is_modern: bool = Field(default=False, description="Modern steel vs historical wooden hull")
    tonnage: Optional[float] = Field(default=None, ge=0, description="Displacement tonnage")
    cargo_mass_category: Literal["light", "medium", "heavy", "precious_metals"] = Field(
        default="medium", description="Cargo weight class"
    )
    buoyancy_factor: float = Field(default=0.35, ge=0.0, le=1.0, description="Reserve buoyancy ratio")
    draft_meters: Optional[float] = Field(default=None, gt=0, description="Draft in meters")
    lkp_lat: Optional[float] = Field(default=None, ge=-90.0, le=90.0, description="LKP latitude")
    lkp_lon: Optional[float] = Field(default=None, ge=-180.0, le=180.0, description="LKP longitude")
    loss_narrative: str = Field(default="", description="Source narrative text")

    def to_vessel_profile(self) -> VesselProfile:
        """Instantiate a calibrated physics VesselProfile using the extracted parameters."""
        profile = profile_from_masts_and_rig(
            mast_count=self.mast_count,
            rig_type=self.rig_type,
            is_modern=self.is_modern,
            tonnage=self.tonnage,
            class_name=self.vessel_name.lower().replace(" ", "_"),
        )
        # If draft was explicitly stated in text, override estimated draft
        if self.draft_meters is not None:
            return VesselProfile(
                class_name=profile.class_name,
                is_modern=profile.is_modern,
                hull_material=profile.hull_material,
                draft_meters=self.draft_meters,
                freeboard_meters=profile.freeboard_meters,
                downwind_leeway_slope=profile.downwind_leeway_slope,
                mast_count=profile.mast_count,
                rig_type=profile.rig_type,
                macro_archetype=profile.macro_archetype,
                downwind_leeway_offset=profile.downwind_leeway_offset,
                crosswind_divergence_deg=profile.crosswind_divergence_deg,
                crosswind_leeway_slope=profile.crosswind_leeway_slope,
                crosswind_leeway_offset=profile.crosswind_leeway_offset,
                jibe_probability_per_hour=profile.jibe_probability_per_hour,
                ballast_ratio=profile.ballast_ratio,
                terminal_descent_rate_m_s=profile.terminal_descent_rate_m_s,
                glide_ratio=profile.glide_ratio,
                structural_fatigue_rate=profile.structural_fatigue_rate,
            )
        return profile


class EpistemicPrior(BaseModel):
    """Calibrated statistical prior covariance derived from archival provenance tiers.

    Attributes:
        source_credibility_tier: 1 (eyewitness log), 2 (fleet sighting), 3 (colonial dispatch).
        provenance_type: Descriptive provenance label.
        spatial_prior_std_nmi: 1-sigma spatial radius uncertainty in nautical miles.
        temporal_prior_std_hours: 1-sigma temporal uncertainty in hours.
        spatial_covariance_matrix_m2: 2x2 diagonal covariance matrix in meters squared.
    """

    source_credibility_tier: Literal[1, 2, 3] = Field(
        default=2, description="Archival reliability tier (1=highest, 3=lowest)"
    )
    provenance_type: str = Field(default="convoy_sighting")
    spatial_prior_std_nmi: float = Field(default=8.0, gt=0)
    temporal_prior_std_hours: float = Field(default=6.0, gt=0)
    spatial_covariance_matrix_m2: List[List[float]] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        """Compute spatial covariance matrix if not provided."""
        if not self.spatial_covariance_matrix_m2:
            std_meters = self.spatial_prior_std_nmi * METERS_PER_NAUTICAL_MILE
            variance = std_meters**2
            self.spatial_covariance_matrix_m2 = [
                [variance, 0.0],
                [0.0, variance],
            ]

    def sample_initial_position(
        self, lkp_lat: float, lkp_lon: float, rng: Optional[np.random.Generator] = None
    ) -> Tuple[float, float]:
        """Sample a perturbed initial particle position around LKP based on spatial variance."""
        gen = rng if rng is not None else np.random.default_rng()
        std_meters = self.spatial_prior_std_nmi * METERS_PER_NAUTICAL_MILE

        # Sample Gaussian offsets in local meters
        dx_m = float(gen.normal(loc=0.0, scale=std_meters))
        dy_m = float(gen.normal(loc=0.0, scale=std_meters))

        # Convert meter offsets to delta degrees
        d_lat = dy_m / 111139.0
        cos_lat = max(0.01, math.cos(math.radians(lkp_lat)))
        d_lon = dx_m / (111139.0 * cos_lat)

        return lkp_lat + d_lat, lkp_lon + d_lon


class JudgeEvaluationRecord(BaseModel):
    """Quality evaluation output produced by Kev / Jev System-One model.

    Attributes:
        is_factual: Boolean indicating whether extracted fields are faithful to source text (Noul).
        confidence_score: Calibrated decision probability score [0.0, 1.0].
        assigned_tier: Categorical credibility tier (Choice).
        hallucination_detected: True if the model flagged invented facts.
        evaluator_name: Name of judge model ('kev', 'jev', 'heuristic').
        notes: Diagnostic feedback.
    """

    is_factual: bool = True
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    assigned_tier: Literal[1, 2, 3] = 2
    hallucination_detected: bool = False
    evaluator_name: str = "kev"
    notes: str = ""
