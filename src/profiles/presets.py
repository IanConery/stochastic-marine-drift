"""Standard calibrated vessel profiles and hierarchical archetype factories."""

import math
from typing import Dict

from src.profiles.vessel_class import MacroArchetype, RigType, VesselProfile

# -------------------------------------------------------------------------
# Macro-Archetype Baseline Profiles (Collapse micro-classes into physical clusters)
# -------------------------------------------------------------------------

ARCHETYPE_1_MAST_FORE_AFT = VesselProfile(
    class_name="archetype_1_mast_fore_aft",
    is_modern=False,
    hull_material="wood",
    mast_count=1,
    rig_type=RigType.FORE_AND_AFT,
    macro_archetype=MacroArchetype.ONE_MAST_FORE_AFT,
    draft_meters=2.2,
    freeboard_meters=1.4,
    downwind_leeway_slope=0.020,
    downwind_leeway_offset=0.02,
    crosswind_divergence_deg=26.0,
    crosswind_leeway_slope=0.012,
    crosswind_leeway_offset=0.01,
    jibe_probability_per_hour=0.15,
    ballast_ratio=0.20,
    terminal_descent_rate_m_s=0.70,
    glide_ratio=0.14,
    structural_fatigue_rate=0.80,
)

ARCHETYPE_2_MAST_FORE_AFT = VesselProfile(
    class_name="archetype_2_mast_fore_aft",
    is_modern=False,
    hull_material="wood",
    mast_count=2,
    rig_type=RigType.FORE_AND_AFT,
    macro_archetype=MacroArchetype.TWO_MAST_FORE_AFT,
    draft_meters=3.2,
    freeboard_meters=2.0,
    downwind_leeway_slope=0.025,
    downwind_leeway_offset=0.03,
    crosswind_divergence_deg=20.0,
    crosswind_leeway_slope=0.009,
    crosswind_leeway_offset=0.01,
    jibe_probability_per_hour=0.08,
    ballast_ratio=0.25,
    terminal_descent_rate_m_s=0.65,
    glide_ratio=0.12,
    structural_fatigue_rate=0.85,
)

ARCHETYPE_2_MAST_SQUARE = VesselProfile(
    class_name="archetype_2_mast_square",
    is_modern=False,
    hull_material="wood",
    mast_count=2,
    rig_type=RigType.SQUARE,
    macro_archetype=MacroArchetype.TWO_MAST_SQUARE,
    draft_meters=3.6,
    freeboard_meters=2.4,
    downwind_leeway_slope=0.032,
    downwind_leeway_offset=0.03,
    crosswind_divergence_deg=6.0,
    crosswind_leeway_slope=0.004,
    crosswind_leeway_offset=0.0,
    jibe_probability_per_hour=0.04,
    ballast_ratio=0.28,
    terminal_descent_rate_m_s=0.62,
    glide_ratio=0.08,
    structural_fatigue_rate=0.95,
)

ARCHETYPE_3_MAST_SQUARE = VesselProfile(
    class_name="archetype_3_mast_square",
    is_modern=False,
    hull_material="wood",
    mast_count=3,
    rig_type=RigType.SQUARE,
    macro_archetype=MacroArchetype.THREE_MAST_SQUARE,
    draft_meters=5.0,
    freeboard_meters=3.2,
    downwind_leeway_slope=0.040,
    downwind_leeway_offset=0.05,
    crosswind_divergence_deg=2.0,
    crosswind_leeway_slope=0.001,
    crosswind_leeway_offset=0.0,
    jibe_probability_per_hour=0.02,
    ballast_ratio=0.33,
    terminal_descent_rate_m_s=0.58,
    glide_ratio=0.06,
    structural_fatigue_rate=1.10,
)

ARCHETYPE_MODERN_UNMASTED = VesselProfile(
    class_name="archetype_modern_unmasted",
    is_modern=True,
    hull_material="steel",
    mast_count=0,
    rig_type=RigType.UNMASTED_POWER,
    macro_archetype=MacroArchetype.MODERN_UNMASTED,
    draft_meters=11.5,
    freeboard_meters=7.5,
    downwind_leeway_slope=0.018,
    downwind_leeway_offset=0.0,
    crosswind_divergence_deg=10.0,
    crosswind_leeway_slope=0.005,
    crosswind_leeway_offset=0.0,
    jibe_probability_per_hour=0.02,
    ballast_ratio=0.15,
    terminal_descent_rate_m_s=3.60,
    glide_ratio=1.15,
    structural_fatigue_rate=0.40,
)

# -------------------------------------------------------------------------
# Calibrated Specific Historical Ship Presets
# -------------------------------------------------------------------------

SPANISH_GALLEON_1620 = VesselProfile(
    class_name="spanish_galleon",
    is_modern=False,
    hull_material="wood",
    mast_count=3,
    rig_type=RigType.SQUARE,
    macro_archetype=MacroArchetype.THREE_MAST_SQUARE,
    draft_meters=4.8,
    freeboard_meters=3.5,
    downwind_leeway_slope=0.042,
    downwind_leeway_offset=0.05,
    crosswind_divergence_deg=0.0,
    crosswind_leeway_slope=0.0,
    crosswind_leeway_offset=0.0,
    jibe_probability_per_hour=0.01,
    ballast_ratio=0.35,
    terminal_descent_rate_m_s=0.55,
    glide_ratio=0.05,
    structural_fatigue_rate=1.20,
)

BERMUDA_SLOOP_1710 = VesselProfile(
    class_name="bermuda_sloop",
    is_modern=False,
    hull_material="wood",
    mast_count=1,
    rig_type=RigType.FORE_AND_AFT,
    macro_archetype=MacroArchetype.ONE_MAST_FORE_AFT,
    draft_meters=2.6,
    freeboard_meters=1.5,
    downwind_leeway_slope=0.022,
    downwind_leeway_offset=0.02,
    crosswind_divergence_deg=28.0,
    crosswind_leeway_slope=0.012,
    crosswind_leeway_offset=0.01,
    jibe_probability_per_hour=0.15,
    ballast_ratio=0.20,
    terminal_descent_rate_m_s=0.70,
    glide_ratio=0.15,
    structural_fatigue_rate=0.80,
)

DUTCH_FLUYT_1680 = VesselProfile(
    class_name="dutch_fluyt",
    is_modern=False,
    hull_material="wood",
    mast_count=3,
    rig_type=RigType.SQUARE,
    macro_archetype=MacroArchetype.THREE_MAST_SQUARE,
    draft_meters=3.8,
    freeboard_meters=2.8,
    downwind_leeway_slope=0.038,
    downwind_leeway_offset=0.04,
    crosswind_divergence_deg=5.0,
    crosswind_leeway_slope=0.003,
    crosswind_leeway_offset=0.0,
    jibe_probability_per_hour=0.03,
    ballast_ratio=0.28,
    terminal_descent_rate_m_s=0.60,
    glide_ratio=0.08,
    structural_fatigue_rate=1.00,
)

HISTORICAL_FRIGATE_1750 = VesselProfile(
    class_name="historical_frigate",
    is_modern=False,
    hull_material="wood",
    mast_count=3,
    rig_type=RigType.SQUARE,
    macro_archetype=MacroArchetype.THREE_MAST_SQUARE,
    draft_meters=5.2,
    freeboard_meters=2.6,
    downwind_leeway_slope=0.032,
    downwind_leeway_offset=0.03,
    crosswind_divergence_deg=8.0,
    crosswind_leeway_slope=0.005,
    crosswind_leeway_offset=0.0,
    jibe_probability_per_hour=0.04,
    ballast_ratio=0.30,
    terminal_descent_rate_m_s=0.65,
    glide_ratio=0.10,
    structural_fatigue_rate=0.90,
)

MODERN_CONTAINER_2015 = VesselProfile(
    class_name="modern_container_ship",
    is_modern=True,
    hull_material="steel",
    mast_count=0,
    rig_type=RigType.UNMASTED_POWER,
    macro_archetype=MacroArchetype.MODERN_UNMASTED,
    draft_meters=12.0,
    freeboard_meters=8.0,
    downwind_leeway_slope=0.018,
    downwind_leeway_offset=0.0,
    crosswind_divergence_deg=12.0,
    crosswind_leeway_slope=0.006,
    crosswind_leeway_offset=0.0,
    jibe_probability_per_hour=0.02,
    ballast_ratio=0.15,
    terminal_descent_rate_m_s=3.80,
    glide_ratio=1.20,
    structural_fatigue_rate=0.40,
)

PRESETS: Dict[str, VesselProfile] = {
    # Named historical presets
    "spanish_galleon": SPANISH_GALLEON_1620,
    "bermuda_sloop": BERMUDA_SLOOP_1710,
    "dutch_fluyt": DUTCH_FLUYT_1680,
    "historical_frigate": HISTORICAL_FRIGATE_1750,
    "modern_container": MODERN_CONTAINER_2015,
    # Macro-archetypes
    "archetype_1_mast_fore_aft": ARCHETYPE_1_MAST_FORE_AFT,
    "archetype_2_mast_fore_aft": ARCHETYPE_2_MAST_FORE_AFT,
    "archetype_2_mast_square": ARCHETYPE_2_MAST_SQUARE,
    "archetype_3_mast_square": ARCHETYPE_3_MAST_SQUARE,
    "archetype_modern_unmasted": ARCHETYPE_MODERN_UNMASTED,
}


def get_vessel_profile(preset_name: str) -> VesselProfile:
    """Retrieve a calibrated vessel profile by preset identifier."""
    if preset_name not in PRESETS:
        valid_keys = list(PRESETS.keys())
        raise KeyError(f"Unknown preset '{preset_name}'. Valid options: {valid_keys}")
    return PRESETS[preset_name]


def profile_from_masts_and_rig(
    mast_count: int,
    rig_type: RigType | str = RigType.SQUARE,
    is_modern: bool = False,
    tonnage: float | None = None,
    class_name: str | None = None,
) -> VesselProfile:
    """Derive physics parameters based on mast count and sail plan orientation.

    This function collapses messy archival descriptions into robust physical macro-archetypes,
    eliminating N=1 data sparsity for rare ship classes while scaling draft and leeway
    continuously if displacement tonnage is known.

    Args:
        mast_count: Number of masts (0 for modern powered vessels).
        rig_type: 'square' or 'fore_and_aft' or RigType enum.
        is_modern: True for steel/powered vessels.
        tonnage: Estimated displacement or burden in tons (optional for dimensional scaling).
        class_name: Optional custom label (e.g. 'armed_snow').

    Returns:
        Calibrated VesselProfile instance.
    """
    if isinstance(rig_type, str):
        rig_type = RigType(rig_type.lower())

    if is_modern or mast_count == 0 or rig_type == RigType.UNMASTED_POWER:
        base = ARCHETYPE_MODERN_UNMASTED
        derived_name = class_name or "modern_unmasted"
        ref_tonnage = 35000.0
    elif mast_count == 1:
        base = ARCHETYPE_1_MAST_FORE_AFT
        derived_name = class_name or "1_mast_fore_aft"
        ref_tonnage = 60.0
    elif mast_count == 2:
        if rig_type == RigType.FORE_AND_AFT:
            base = ARCHETYPE_2_MAST_FORE_AFT
            derived_name = class_name or "2_mast_fore_aft"
            ref_tonnage = 140.0
        else:
            base = ARCHETYPE_2_MAST_SQUARE
            derived_name = class_name or "2_mast_square"
            ref_tonnage = 220.0
    else:  # mast_count >= 3
        base = ARCHETYPE_3_MAST_SQUARE
        derived_name = class_name or f"{mast_count}_mast_square"
        ref_tonnage = 650.0

    # If tonnage is provided, apply continuous volumetric scaling (Draft ~ Tonnage^(1/3))
    draft = base.draft_meters
    freeboard = base.freeboard_meters
    downwind_slope = base.downwind_leeway_slope

    if tonnage is not None and tonnage > 0:
        scale_factor = (tonnage / ref_tonnage) ** (1.0 / 3.0)
        # Constrain scaling to realistic physical limits [0.6, 1.8]
        scale_factor = max(0.6, min(1.8, scale_factor))
        draft = round(base.draft_meters * scale_factor, 2)
        freeboard = round(base.freeboard_meters * (scale_factor ** 0.8), 2)
        # Heavier hulls of same rig have slightly more inertia and deeper grip on water
        downwind_slope = round(max(0.010, min(0.080, base.downwind_leeway_slope / (scale_factor ** 0.2))), 4)

    return VesselProfile(
        class_name=derived_name,
        is_modern=base.is_modern,
        hull_material=base.hull_material,
        mast_count=mast_count,
        rig_type=rig_type,
        macro_archetype=base.macro_archetype,
        draft_meters=draft,
        freeboard_meters=freeboard,
        downwind_leeway_slope=downwind_slope,
        downwind_leeway_offset=base.downwind_leeway_offset,
        crosswind_divergence_deg=base.crosswind_divergence_deg,
        crosswind_leeway_slope=base.crosswind_leeway_slope,
        crosswind_leeway_offset=base.crosswind_leeway_offset,
        jibe_probability_per_hour=base.jibe_probability_per_hour,
        ballast_ratio=base.ballast_ratio,
        terminal_descent_rate_m_s=base.terminal_descent_rate_m_s,
        glide_ratio=base.glide_ratio,
        structural_fatigue_rate=base.structural_fatigue_rate,
    )
