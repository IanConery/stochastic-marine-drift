"""Vessel profiles, leeway taxonomy, and archetype factories."""

from src.profiles.presets import (
    ARCHETYPE_1_MAST_FORE_AFT,
    ARCHETYPE_2_MAST_FORE_AFT,
    ARCHETYPE_2_MAST_SQUARE,
    ARCHETYPE_3_MAST_SQUARE,
    ARCHETYPE_MODERN_UNMASTED,
    BERMUDA_SLOOP_1710,
    DUTCH_FLUYT_1680,
    HISTORICAL_FRIGATE_1750,
    MODERN_CONTAINER_2015,
    PRESETS,
    SPANISH_GALLEON_1620,
    get_vessel_profile,
    profile_from_masts_and_rig,
)
from src.profiles.vessel_class import MacroArchetype, RigType, VesselProfile

__all__ = [
    "VesselProfile",
    "RigType",
    "MacroArchetype",
    "SPANISH_GALLEON_1620",
    "BERMUDA_SLOOP_1710",
    "DUTCH_FLUYT_1680",
    "HISTORICAL_FRIGATE_1750",
    "MODERN_CONTAINER_2015",
    "ARCHETYPE_1_MAST_FORE_AFT",
    "ARCHETYPE_2_MAST_FORE_AFT",
    "ARCHETYPE_2_MAST_SQUARE",
    "ARCHETYPE_3_MAST_SQUARE",
    "ARCHETYPE_MODERN_UNMASTED",
    "PRESETS",
    "get_vessel_profile",
    "profile_from_masts_and_rig",
]
