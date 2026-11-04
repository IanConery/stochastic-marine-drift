"""Vessel taxonomy and hydrodynamic parameter definitions for Stochastic Marine Drift."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class RigType(str, Enum):
    """Primary aerodynamic sail plan orientation."""

    SQUARE = "square"
    FORE_AND_AFT = "fore_and_aft"
    UNMASTED_POWER = "unmasted_power"


class MacroArchetype(str, Enum):
    """Macro-archetype categories collapsing historical micro-classes into physical clusters."""

    ONE_MAST_FORE_AFT = "1_mast_fore_aft"
    TWO_MAST_FORE_AFT = "2_mast_fore_aft"
    TWO_MAST_SQUARE = "2_mast_square"
    THREE_MAST_SQUARE = "3_mast_square"
    MODERN_UNMASTED = "modern_unmasted"


@dataclass(frozen=True)
class VesselProfile:
    """Defines hydrodynamic, leeway, and structural characteristics for a vessel class.

    Attributes:
        class_name: Identifier for the vessel type (e.g., 'spanish_galleon').
        is_modern: False for historical timber/sail; True for modern powered/steel.
        hull_material: Hull composition ('wood', 'iron', 'steel').
        mast_count: Number of standing masts (0 for modern powered vessels).
        rig_type: Sail plan ('square', 'fore_and_aft', 'unmasted_power').
        macro_archetype: Physical clustering category.
        draft_meters: Distance from waterline to bottom of keel.
        freeboard_meters: Height of reserve buoyancy from waterline to main deck.
        downwind_leeway_slope: Multiplier for wind speed along downwind vector (a_dw).
        downwind_leeway_offset: Additive leeway speed offset in m/s (b_dw).
        crosswind_divergence_deg: Absolute divergence angle off downwind vector.
        crosswind_leeway_slope: Multiplier for crosswind leeway speed (a_cw).
        crosswind_leeway_offset: Additive crosswind leeway offset in m/s (b_cw).
        jibe_probability_per_hour: Hourly probability of shifting crosswind tack.
        ballast_ratio: Ballast mass as a fraction of total displacement.
        terminal_descent_rate_m_s: Vertical sinking velocity through water column.
        glide_ratio: Horizontal displacement per unit of vertical descent (L/D).
        structural_fatigue_rate: Hull failure sensitivity under wave impact stress.
    """

    class_name: str
    is_modern: bool
    hull_material: Literal["wood", "iron", "steel"]
    draft_meters: float
    freeboard_meters: float
    downwind_leeway_slope: float
    mast_count: int = 3
    rig_type: RigType = RigType.SQUARE
    macro_archetype: MacroArchetype | None = None
    downwind_leeway_offset: float = 0.0
    crosswind_divergence_deg: float = 0.0
    crosswind_leeway_slope: float = 0.0
    crosswind_leeway_offset: float = 0.0
    jibe_probability_per_hour: float = 0.05
    ballast_ratio: float = 0.25
    terminal_descent_rate_m_s: float = 0.65
    glide_ratio: float = 0.10
    structural_fatigue_rate: float = 1.0

    def __post_init__(self) -> None:
        """Validate physical parameters."""
        if self.mast_count < 0:
            raise ValueError(f"mast_count cannot be negative, got {self.mast_count}")
        if self.draft_meters <= 0:
            raise ValueError(f"draft_meters must be positive, got {self.draft_meters}")
        if self.freeboard_meters <= 0:
            raise ValueError(f"freeboard_meters must be positive, got {self.freeboard_meters}")
        if not (0.005 <= self.downwind_leeway_slope <= 0.10):
            raise ValueError(
                f"downwind_leeway_slope must be in [0.005, 0.10], got {self.downwind_leeway_slope}"
            )
        if not (0.0 <= self.crosswind_divergence_deg <= 60.0):
            raise ValueError(
                f"crosswind_divergence_deg must be in [0, 60], got {self.crosswind_divergence_deg}"
            )
        if self.terminal_descent_rate_m_s <= 0:
            raise ValueError(
                f"terminal_descent_rate_m_s must be positive, got {self.terminal_descent_rate_m_s}"
            )
        if self.glide_ratio < 0:
            raise ValueError(f"glide_ratio cannot be negative, got {self.glide_ratio}")
