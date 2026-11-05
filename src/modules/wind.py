"""Atmospheric vector updates, Weibull stochastic turbulence, and coastal land-sea breeze."""

import math
from typing import Optional, Tuple
import numpy as np


def wind_speed_and_direction(u10: float, v10: float) -> Tuple[float, float]:
    """Convert eastward (u) and northward (v) wind components to speed and direction.

    Direction is returned in oceanographic convention (direction towards which wind blows,
    measured clockwise from True North in degrees [0, 360)).

    Args:
        u10: Eastward wind component in m/s.
        v10: Northward wind component in m/s.

    Returns:
        Tuple of (speed_m_s, direction_deg_to).
    """
    speed = math.hypot(u10, v10)
    if speed == 0.0:
        return 0.0, 0.0

    # math.atan2(u, v) yields angle clockwise from North (v axis)
    direction_rad = math.atan2(u10, v10)
    direction_deg = math.degrees(direction_rad) % 360.0
    return speed, direction_deg


def wind_components_from_speed_dir(speed: float, direction_deg_to: float) -> Tuple[float, float]:
    """Convert speed and direction (towards which wind blows) to (u, v) components.

    Args:
        speed: Wind speed in m/s (must be >= 0).
        direction_deg_to: Heading clockwise from True North in degrees.

    Returns:
        Tuple of (u10, v10) in m/s.
    """
    if speed < 0:
        raise ValueError(f"speed cannot be negative, got {speed}")
    rad = math.radians(direction_deg_to % 360.0)
    u = speed * math.sin(rad)
    v = speed * math.cos(rad)
    return u, v


def sample_weibull_turbulence(
    mean_speed: float,
    weibull_shape: float = 2.0,
    turbulence_intensity: float = 0.15,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Sample stochastic wind speed fluctuation using a normalized Weibull distribution.

    The Weibull multiplier is normalized such that its expectation E[multiplier] = 1.0.
    The resulting speed is:
        speed_turbulent = mean_speed * (1.0 + turbulence_intensity * (multiplier - 1.0))

    Args:
        mean_speed: Baseline wind speed in m/s.
        weibull_shape: Shape parameter k (typically 1.8 - 2.5 for boundary layer winds).
        turbulence_intensity: Fractional turbulence intensity (standard deviation / mean).
        rng: Optional numpy Generator for reproducibility.

    Returns:
        Turbulent instantaneous wind speed in m/s.
    """
    if mean_speed <= 0.0:
        return 0.0
    if weibull_shape <= 0.0:
        raise ValueError(f"weibull_shape must be positive, got {weibull_shape}")

    gen = rng if rng is not None else np.random.default_rng()
    # Scipy/Numpy weibull samples from standard 1-parameter Weibull (scale=1)
    # E[X] = Gamma(1 + 1/k)
    raw_sample = gen.weibull(weibull_shape)
    gamma_mean = math.gamma(1.0 + (1.0 / weibull_shape))
    normalized_sample = raw_sample / gamma_mean

    multiplier = 1.0 + turbulence_intensity * (normalized_sample - 1.0)
    # Wind speed cannot become negative
    return max(0.0, mean_speed * multiplier)


def compute_land_sea_breeze_perturbation(
    dist_to_coast_km: float,
    coast_onshore_heading_deg: float,
    local_solar_hour: float,
    max_breeze_speed_m_s: float = 4.5,
    decay_scale_km: float = 30.0,
) -> Tuple[float, float]:
    """Calculate diurnal land-sea breeze vector offset near high-relief island coasts.

    In the Caribbean (Cuba, Hispaniola, Jamaica, Puerto Rico), intense daytime heating
    drives onshore winds (peaking ~15:00 local time), while nighttime radiative cooling
    drives offshore catabatic breezes (peaking ~03:00 local time).

    Args:
        dist_to_coast_km: Distance from particle to nearest island coastline in km.
        coast_onshore_heading_deg: Compass heading perpendicular to coast pointing inland (degrees).
        local_solar_hour: Local time of day [0.0, 24.0).
        max_breeze_speed_m_s: Maximum sea breeze speed at shoreline in m/s.
        decay_scale_km: Characteristic exponential decay length offshore in km.

    Returns:
        Tuple of (u_breeze, v_breeze) in m/s to be added to synoptic wind.
    """
    if dist_to_coast_km < 0:
        dist_to_coast_km = 0.0

    # Diurnal cycle: peak onshore (+1) at 15:00, peak offshore (-1) at 03:00
    # Phase shift: sin(2 * pi * (hour - 9.0) / 24.0)
    hour_norm = (local_solar_hour % 24.0)
    cycle_factor = math.sin(2.0 * math.pi * (hour_norm - 9.0) / 24.0)

    # Exponential decay offshore
    spatial_factor = math.exp(-dist_to_coast_km / decay_scale_km)
    breeze_magnitude = max_breeze_speed_m_s * cycle_factor * spatial_factor

    # Onshore heading: positive breeze_magnitude pushes inland (+coast_onshore_heading_deg)
    onshore_rad = math.radians(coast_onshore_heading_deg % 360.0)
    u_breeze = breeze_magnitude * math.sin(onshore_rad)
    v_breeze = breeze_magnitude * math.cos(onshore_rad)
    return u_breeze, v_breeze


def perturb_wind_vector(
    u10: float,
    v10: float,
    dist_to_coast_km: Optional[float] = None,
    coast_onshore_heading_deg: Optional[float] = None,
    local_solar_hour: Optional[float] = None,
    weibull_shape: float = 2.0,
    turbulence_intensity: float = 0.15,
    direction_noise_std_deg: float = 8.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Generate a stochastically perturbed wind vector integrating Weibull turbulence and coastal breeze.

    Args:
        u10: Baseline eastward wind component in m/s.
        v10: Baseline northward wind component in m/s.
        dist_to_coast_km: Distance to coast in km (if near island).
        coast_onshore_heading_deg: Normal vector heading pointing inland (degrees).
        local_solar_hour: Hour of day in local time [0, 24).
        weibull_shape: Weibull shape parameter k for gust speed.
        turbulence_intensity: Fractional turbulence intensity.
        direction_noise_std_deg: Gaussian standard deviation for turbulent directional veer.
        rng: Optional numpy Generator for reproducibility.

    Returns:
        Tuple of (u_perturbed, v_perturbed) in m/s.
    """
    gen = rng if rng is not None else np.random.default_rng()

    # 1. Decompose into baseline speed and direction
    base_speed, base_dir = wind_speed_and_direction(u10, v10)
    if base_speed == 0.0:
        return 0.0, 0.0

    # 2. Apply Weibull turbulence to speed and Gaussian noise to direction
    turb_speed = sample_weibull_turbulence(
        base_speed, weibull_shape=weibull_shape, turbulence_intensity=turbulence_intensity, rng=gen
    )
    dir_offset = gen.normal(loc=0.0, scale=direction_noise_std_deg)
    turb_dir = (base_dir + dir_offset) % 360.0

    u_synoptic, v_synoptic = wind_components_from_speed_dir(turb_speed, turb_dir)

    # 3. Add coastal diurnal land-sea breeze if coastal parameters provided
    if (
        dist_to_coast_km is not None
        and coast_onshore_heading_deg is not None
        and local_solar_hour is not None
    ):
        u_breeze, v_breeze = compute_land_sea_breeze_perturbation(
            dist_to_coast_km=dist_to_coast_km,
            coast_onshore_heading_deg=coast_onshore_heading_deg,
            local_solar_hour=local_solar_hour,
        )
        return u_synoptic + u_breeze, v_synoptic + v_breeze

    return u_synoptic, v_synoptic
