"""Decoupled environmental physics modules for Stochastic Marine Drift."""

from src.modules.current import (
    compute_ekman_surface_velocity,
    compute_total_surface_drift_current,
    coriolis_parameter,
    drag_coefficient_wu,
    interpolate_current_with_depth,
)
from src.modules.deep_water import (
    compute_terminal_descent_velocity,
    simulate_3d_ascent_backtrack,
    simulate_3d_descent,
)
from src.modules.extreme_event import (
    TropicalCyclone,
    haversine_distance_and_bearing,
)
from src.modules.stokes import (
    compute_stokes_drift_vector,
    compute_stokes_profile,
    compute_surface_stokes_speed,
)
from src.modules.wind import (
    compute_land_sea_breeze_perturbation,
    perturb_wind_vector,
    sample_weibull_turbulence,
    wind_components_from_speed_dir,
    wind_speed_and_direction,
)

__all__ = [
    # Stokes
    "compute_surface_stokes_speed",
    "compute_stokes_drift_vector",
    "compute_stokes_profile",
    # Wind
    "wind_speed_and_direction",
    "wind_components_from_speed_dir",
    "sample_weibull_turbulence",
    "compute_land_sea_breeze_perturbation",
    "perturb_wind_vector",
    # Current
    "coriolis_parameter",
    "drag_coefficient_wu",
    "compute_ekman_surface_velocity",
    "interpolate_current_with_depth",
    "compute_total_surface_drift_current",
    # Extreme events
    "TropicalCyclone",
    "haversine_distance_and_bearing",
    # Deep water
    "compute_terminal_descent_velocity",
    "simulate_3d_descent",
    "simulate_3d_ascent_backtrack",
]
