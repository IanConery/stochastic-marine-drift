"""Stokes wave drift solver for wave-induced net mass transport."""

import math
from typing import Literal, Tuple


def compute_surface_stokes_speed(
    wave_height_m: float,
    peak_period_s: float,
    gravity: float = 9.80665,
) -> float:
    """Compute surface Stokes drift speed in deep water.

    Uses deep-water monochromatic approximation:
        U_s(0) = (pi^3 * H_s^2) / (2 * g * T_p^3)

    Args:
        wave_height_m: Significant wave height (H_s) in meters. Must be >= 0.
        peak_period_s: Peak wave period (T_p) in seconds. Must be > 0.
        gravity: Gravitational acceleration (m/s^2). Default is 9.80665.

    Returns:
        Surface Stokes drift speed in meters per second (m/s).

    Raises:
        ValueError: If wave_height_m < 0 or peak_period_s <= 0.
    """
    if wave_height_m < 0:
        raise ValueError(f"wave_height_m cannot be negative, got {wave_height_m}")
    if peak_period_s <= 0:
        raise ValueError(f"peak_period_s must be strictly positive, got {peak_period_s}")
    if wave_height_m == 0.0:
        return 0.0

    return (math.pi**3 * (wave_height_m**2)) / (2.0 * gravity * (peak_period_s**3))


def compute_stokes_drift_vector(
    wave_height_m: float,
    peak_period_s: float,
    wave_direction_deg: float,
    direction_convention: Literal["to", "from"] = "to",
    gravity: float = 9.80665,
) -> Tuple[float, float]:
    """Compute Stokes drift vector components (East, North) in m/s.

    Args:
        wave_height_m: Significant wave height (H_s) in meters.
        peak_period_s: Peak wave period (T_p) in seconds.
        wave_direction_deg: Wave propagation heading in degrees [0, 360).
            0 = North, 90 = East, 180 = South, 270 = West.
        direction_convention: 'to' if heading is direction of wave travel (oceanographic),
            'from' if heading is direction waves come from (meteorological).
        gravity: Gravitational acceleration in m/s^2.

    Returns:
        Tuple of (u_stokes, v_stokes) in meters per second.
    """
    speed = compute_surface_stokes_speed(wave_height_m, peak_period_s, gravity=gravity)
    if speed == 0.0:
        return 0.0, 0.0

    heading_rad = math.radians(wave_direction_deg % 360.0)
    if direction_convention == "from":
        # Wave moving opposite to the source direction
        heading_rad = (heading_rad + math.pi) % (2.0 * math.pi)

    # Standard navigational angle: 0 = North (+v), 90 = East (+u)
    u_stokes = speed * math.sin(heading_rad)
    v_stokes = speed * math.cos(heading_rad)
    return u_stokes, v_stokes


def compute_stokes_profile(
    wave_height_m: float,
    peak_period_s: float,
    depth_m: float,
    gravity: float = 9.80665,
) -> float:
    """Compute Stokes drift speed at an arbitrary subsurface depth.

    Stokes drift decays exponentially with depth:
        U_s(z) = U_s(0) * exp(2 * k * z)
    where k = (4 * pi^2) / (g * T_p^2) is the deep-water wavenumber and z <= 0.

    Args:
        wave_height_m: Significant wave height (H_s) in meters.
        peak_period_s: Peak wave period (T_p) in seconds.
        depth_m: Water depth below surface (positive downwards, e.g. 5.0 for 5m depth).
        gravity: Gravitational acceleration in m/s^2.

    Returns:
        Stokes drift speed at specified depth in m/s.
    """
    if depth_m < 0:
        raise ValueError(f"depth_m must be non-negative (distance below surface), got {depth_m}")

    surface_speed = compute_surface_stokes_speed(wave_height_m, peak_period_s, gravity=gravity)
    if surface_speed == 0.0 or depth_m == 0.0:
        return surface_speed

    # Deep-water wavenumber k
    k = (4.0 * (math.pi**2)) / (gravity * (peak_period_s**2))
    # Decay factor exp(-2 * k * depth)
    decay = math.exp(-2.0 * k * depth_m)
    return surface_speed * decay
