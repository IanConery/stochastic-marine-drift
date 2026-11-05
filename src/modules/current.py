"""Ocean current velocity interpolation, Ekman surface layer shear, and multi-depth profiles."""

import math
from typing import Tuple

# Earth's angular velocity (rad/s)
OMEGA_EARTH = 7.2921159e-5
# Standard seawater density (kg/m^3)
RHO_SEAWATER = 1025.0
# Standard air density at sea level (kg/m^3)
RHO_AIR = 1.225


def coriolis_parameter(latitude_deg: float) -> float:
    """Calculate the Coriolis frequency f = 2 * Omega * sin(phi) in s^-1.

    Args:
        latitude_deg: Latitude in degrees [-90, 90].

    Returns:
        Coriolis frequency f in radians per second (positive in Northern Hemisphere).
    """
    lat_rad = math.radians(latitude_deg)
    return 2.0 * OMEGA_EARTH * math.sin(lat_rad)


def drag_coefficient_wu(wind_speed_m_s: float) -> float:
    """Compute 10m wind drag coefficient C_d using the Wu (1982) formulation.

    C_d = (0.8 + 0.065 * U_10) * 1e-3, bounded between 1.0e-3 and 3.0e-3.

    Args:
        wind_speed_m_s: Wind speed in m/s.

    Returns:
        Dimensionless air-sea drag coefficient.
    """
    speed = max(0.0, wind_speed_m_s)
    cd = (0.8 + 0.065 * speed) * 1e-3
    return min(max(cd, 1.0e-3), 3.0e-3)


def compute_ekman_surface_velocity(
    u10: float,
    v10: float,
    latitude_deg: float,
    vertical_eddy_viscosity: float = 0.02,
) -> Tuple[float, float]:
    """Calculate wind-driven Ekman surface drift velocity in m/s.

    In the Northern Hemisphere (including the entire Caribbean Basin), surface Ekman
    transport is deflected 45 degrees to the right of the surface wind vector.

    Args:
        u10: Eastward wind component in m/s.
        v10: Northward wind component in m/s.
        latitude_deg: Latitude in degrees (must be non-equatorial, |lat| >= 2.0).
        vertical_eddy_viscosity: Vertical turbulent eddy viscosity A_z in m^2/s (~0.01 - 0.05).

    Returns:
        Tuple of (u_ekman, v_ekman) in m/s.
    """
    wind_speed = math.hypot(u10, v10)
    if wind_speed == 0.0:
        return 0.0, 0.0

    f = coriolis_parameter(latitude_deg)
    if abs(f) < 1.0e-6:
        # Near equator (|lat| < 0.4°), Ekman balance breaks down; return zero
        return 0.0, 0.0

    # Wind stress tau = rho_air * C_d * W^2
    cd = drag_coefficient_wu(wind_speed)
    tau = RHO_AIR * cd * (wind_speed**2)

    # Surface Ekman speed V_0 = tau / (rho_w * sqrt(|f| * A_z))
    denom = RHO_SEAWATER * math.sqrt(abs(f) * vertical_eddy_viscosity)
    v0 = tau / denom
    # Physical cap: Ekman drift rarely exceeds 3.5% of wind speed
    v0 = min(v0, 0.035 * wind_speed)

    # Wind angle: angle clockwise from North towards which wind blows
    wind_dir_rad = math.atan2(u10, v10)

    # Deflection angle: +45 degrees (pi/4) in Northern Hemisphere, -45 degrees in Southern
    deflection_rad = math.pi / 4.0 if f > 0 else -math.pi / 4.0
    ekman_dir_rad = wind_dir_rad + deflection_rad

    u_ekman = v0 * math.sin(ekman_dir_rad)
    v_ekman = v0 * math.cos(ekman_dir_rad)
    return u_ekman, v_ekman


def interpolate_current_with_depth(
    depth_m: float,
    u_surface: float,
    v_surface: float,
    u_deep: float,
    v_deep: float,
    deep_reference_depth_m: float = 100.0,
) -> Tuple[float, float]:
    """Interpolate horizontal ocean current between surface and deep reference layers.

    Uses smooth exponential decay transitioning from surface current to deep column.

    Args:
        depth_m: Current depth below surface (positive down, e.g. 30.0 for 30m depth).
        u_surface: Eastward velocity at surface (m/s).
        v_surface: Northward velocity at surface (m/s).
        u_deep: Eastward velocity at deep reference layer (m/s).
        v_deep: Northward velocity at deep reference layer (m/s).
        deep_reference_depth_m: Depth at which deep current dominates in meters.

    Returns:
        Tuple of (u_current, v_current) at specified depth in m/s.
    """
    if depth_m <= 0.0:
        return u_surface, v_surface
    if deep_reference_depth_m <= 0.0:
        raise ValueError("deep_reference_depth_m must be strictly positive")

    # Weight factor w: 1 at surface, 0 at deep_reference_depth_m
    norm_depth = min(1.0, depth_m / deep_reference_depth_m)
    # Cosine taper for smooth transition
    weight_surface = 0.5 * (1.0 + math.cos(math.pi * norm_depth))
    weight_deep = 1.0 - weight_surface

    u = weight_surface * u_surface + weight_deep * u_deep
    v = weight_surface * v_surface + weight_deep * v_deep
    return u, v


def compute_total_surface_drift_current(
    u_ocean_geostrophic: float,
    v_ocean_geostrophic: float,
    u10_wind: float,
    v10_wind: float,
    latitude_deg: float,
    include_ekman: bool = True,
) -> Tuple[float, float]:
    """Compute total effective surface current combining geostrophic/tidal ocean vectors with Ekman drift.

    Args:
        u_ocean_geostrophic: Background ocean model Eastward current in m/s.
        v_ocean_geostrophic: Background ocean model Northward current in m/s.
        u10_wind: 10m Eastward wind component in m/s.
        v10_wind: 10m Northward wind component in m/s.
        latitude_deg: Latitude in degrees.
        include_ekman: Whether to compute and add surface Ekman deflection layer.

    Returns:
        Tuple of (u_total, v_total) current vectors in m/s.
    """
    u_tot = u_ocean_geostrophic
    v_tot = v_ocean_geostrophic

    if include_ekman:
        u_ek, v_ek = compute_ekman_surface_velocity(u10_wind, v10_wind, latitude_deg=latitude_deg)
        u_tot += u_ek
        v_tot += v_ek

    return u_tot, v_tot
