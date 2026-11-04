"""Unit tests for vessel profiles, validation logic, and mast/rig archetype derivation."""

import pytest
from src.profiles.presets import (
    ARCHETYPE_1_MAST_FORE_AFT,
    ARCHETYPE_2_MAST_FORE_AFT,
    ARCHETYPE_2_MAST_SQUARE,
    ARCHETYPE_3_MAST_SQUARE,
    ARCHETYPE_MODERN_UNMASTED,
    BERMUDA_SLOOP_1710,
    MODERN_CONTAINER_2015,
    PRESETS,
    SPANISH_GALLEON_1620,
    get_vessel_profile,
    profile_from_masts_and_rig,
)
from src.profiles.vessel_class import MacroArchetype, RigType, VesselProfile


def test_spanish_galleon_parameters():
    galleon = SPANISH_GALLEON_1620
    assert galleon.class_name == "spanish_galleon"
    assert not galleon.is_modern
    assert galleon.hull_material == "wood"
    assert galleon.mast_count == 3
    assert galleon.rig_type == RigType.SQUARE
    assert galleon.macro_archetype == MacroArchetype.THREE_MAST_SQUARE
    assert galleon.draft_meters == 4.8
    assert galleon.downwind_leeway_slope == 0.042
    assert galleon.crosswind_divergence_deg == 0.0
    assert galleon.glide_ratio == 0.05


def test_bermuda_sloop_crosswind():
    sloop = BERMUDA_SLOOP_1710
    assert sloop.mast_count == 1
    assert sloop.rig_type == RigType.FORE_AND_AFT
    assert sloop.crosswind_divergence_deg == 28.0
    assert sloop.jibe_probability_per_hour > SPANISH_GALLEON_1620.jibe_probability_per_hour


def test_modern_container_glide():
    bulker = MODERN_CONTAINER_2015
    assert bulker.is_modern
    assert bulker.mast_count == 0
    assert bulker.hull_material == "steel"
    assert bulker.terminal_descent_rate_m_s > 3.0
    assert bulker.glide_ratio >= 1.0


def test_invalid_draft_raises():
    with pytest.raises(ValueError, match="draft_meters must be positive"):
        VesselProfile(
            class_name="invalid",
            is_modern=False,
            hull_material="wood",
            draft_meters=-1.0,
            freeboard_meters=2.0,
            downwind_leeway_slope=0.03,
        )


def test_invalid_windage_raises():
    with pytest.raises(ValueError, match="downwind_leeway_slope"):
        VesselProfile(
            class_name="invalid",
            is_modern=False,
            hull_material="wood",
            draft_meters=3.0,
            freeboard_meters=2.0,
            downwind_leeway_slope=0.25,  # Exceeds realistic physical bounds
        )


def test_invalid_mast_count_raises():
    with pytest.raises(ValueError, match="mast_count cannot be negative"):
        VesselProfile(
            class_name="invalid",
            is_modern=False,
            hull_material="wood",
            draft_meters=3.0,
            freeboard_meters=2.0,
            downwind_leeway_slope=0.03,
            mast_count=-1,
        )


def test_get_vessel_profile_lookup():
    profile = get_vessel_profile("spanish_galleon")
    assert profile == SPANISH_GALLEON_1620

    with pytest.raises(KeyError, match="Unknown preset"):
        get_vessel_profile("spaceship")


def test_profile_from_masts_and_rig_one_mast():
    profile = profile_from_masts_and_rig(mast_count=1, rig_type=RigType.FORE_AND_AFT)
    assert profile.macro_archetype == MacroArchetype.ONE_MAST_FORE_AFT
    assert profile.crosswind_divergence_deg == 26.0
    assert profile.draft_meters == ARCHETYPE_1_MAST_FORE_AFT.draft_meters


def test_profile_from_masts_and_rig_two_mast_distinction():
    # 2-mast fore-and-aft (schooner) vs 2-mast square (brig)
    schooner = profile_from_masts_and_rig(mast_count=2, rig_type=RigType.FORE_AND_AFT)
    brig = profile_from_masts_and_rig(mast_count=2, rig_type=RigType.SQUARE)

    assert schooner.macro_archetype == MacroArchetype.TWO_MAST_FORE_AFT
    assert brig.macro_archetype == MacroArchetype.TWO_MAST_SQUARE
    assert schooner.crosswind_divergence_deg > brig.crosswind_divergence_deg
    assert brig.downwind_leeway_slope > schooner.downwind_leeway_slope


def test_profile_from_masts_and_rig_three_mast_heavy():
    ship = profile_from_masts_and_rig(mast_count=3, rig_type="square")
    assert ship.macro_archetype == MacroArchetype.THREE_MAST_SQUARE
    assert ship.draft_meters == ARCHETYPE_3_MAST_SQUARE.draft_meters
    assert ship.downwind_leeway_slope == ARCHETYPE_3_MAST_SQUARE.downwind_leeway_slope


def test_profile_from_masts_and_rig_modern():
    modern = profile_from_masts_and_rig(mast_count=0, is_modern=True)
    assert modern.macro_archetype == MacroArchetype.MODERN_UNMASTED
    assert modern.is_modern
    assert modern.terminal_descent_rate_m_s > 3.0


def test_profile_tonnage_scaling():
    # A heavy 1200-ton 3-masted ship vs standard 650-ton archetype
    heavy_ship = profile_from_masts_and_rig(
        mast_count=3, rig_type=RigType.SQUARE, tonnage=1200.0, class_name="heavy_warship"
    )
    std_ship = ARCHETYPE_3_MAST_SQUARE

    assert heavy_ship.class_name == "heavy_warship"
    assert heavy_ship.draft_meters > std_ship.draft_meters
    assert heavy_ship.freeboard_meters > std_ship.freeboard_meters
