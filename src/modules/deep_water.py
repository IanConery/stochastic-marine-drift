"""3D water column physics, terminal descent/ascent velocity, and hydrodynamic glide."""

import math
from typing import Callable, List, Optional, Tuple

from src.profiles.vessel_class import VesselProfile


def compute_terminal_descent_velocity(
    displacement_volume_m3: float,
    vessel_mass_kg: float,
    projected_area_m2: float,
    drag_coefficient: float = 1.2,
    seawater_density_kg_m3: float = 1025.0,
    gravity: float = 9.80665,
) -> float:
    """Compute terminal vertical sinking velocity w_term from hydrodynamics.

    Balancing negative buoyancy against quadratic hydrodynamic drag:
        F_net = (vessel_mass * g) - (seawater_density * displacement_volume * g)
        F_drag = 0.5 * seawater_density * drag_coefficient * projected_area * w_term^2

    Args:
        displacement_volume_m3: Volume of displaced water in m^3 (accounting for flooded holds).
        vessel_mass_kg: Total vessel mass including ballast and cargo in kg.
        projected_area_m2: Cross-sectional projected drag area in m^2.
        drag_coefficient: Dimensionless drag coefficient C_d (~1.0 - 1.4 for tumbling wreckage).
        seawater_density_kg_m3: Water density in kg/m^3 (~1025 kg/m^3).
        gravity: Gravitational acceleration in m/s^2.

    Returns:
        Terminal descent velocity in m/s (positive downwards).

    Raises:
        ValueError: If vessel is positively buoyant (will not sink) or projected_area <= 0.
    """
    if projected_area_m2 <= 0:
        raise ValueError("projected_area_m2 must be strictly positive")

    buoyant_force = seawater_density_kg_m3 * displacement_volume_m3 * gravity
    gravitational_force = vessel_mass_kg * gravity
    net_downward_force = gravitational_force - buoyant_force

    if net_downward_force <= 0:
        raise ValueError(
            "Vessel is positively buoyant (net downward force <= 0); it will not sink."
        )

    denom = 0.5 * seawater_density_kg_m3 * drag_coefficient * projected_area_m2
    return math.sqrt(net_downward_force / denom)


def simulate_3d_descent(
    sps_x_m: float,
    sps_y_m: float,
    seafloor_depth_m: float,
    profile: VesselProfile,
    current_field_fn: Optional[Callable[[float, float, float], Tuple[float, float]]] = None,
    glide_heading_deg: Optional[float] = None,
    num_depth_steps: int = 50,
) -> Tuple[float, float, float, List[Tuple[float, float, float]]]:
    """Simulate 3D underwater descent from Surface Point of Sink (SPS) to seafloor.

    Integrates multi-depth horizontal current deflection and horizontal hydrodynamic glide.

    Args:
        sps_x_m: Eastward surface coordinate in meters.
        sps_y_m: Northward surface coordinate in meters.
        seafloor_depth_m: Total water column depth in meters (positive value, e.g. 800.0).
        profile: VesselProfile providing terminal descent velocity and glide ratio.
        current_field_fn: Optional function (x_m, y_m, depth_m) -> (u_current_m_s, v_current_m_s).
        glide_heading_deg: Heading angle (degrees clockwise from North) of hydrodynamic glide.
        num_depth_steps: Number of vertical integration layers.

    Returns:
        Tuple of:
            - seafloor_x_m: Final Eastward resting coordinate on bottom.
            - seafloor_y_m: Final Northward resting coordinate on bottom.
            - total_descent_time_s: Duration of descent in seconds.
            - trajectory_waypoints: List of (x_m, y_m, z_depth_m) points during descent.
    """
    if seafloor_depth_m <= 0:
        raise ValueError(f"seafloor_depth_m must be strictly positive, got {seafloor_depth_m}")

    w_term = profile.terminal_descent_rate_m_s
    total_descent_time_s = seafloor_depth_m / w_term

    dz = seafloor_depth_m / float(num_depth_steps)
    dt = dz / w_term

    cur_x = sps_x_m
    cur_y = sps_y_m
    cur_z = 0.0

    waypoints: List[Tuple[float, float, float]] = [(cur_x, cur_y, cur_z)]

    # Compute step glide vector if glide ratio > 0
    glide_ratio = profile.glide_ratio
    heading_rad = (
        math.radians(glide_heading_deg % 360.0)
        if glide_heading_deg is not None
        else 0.0
    )
    glide_vx = (glide_ratio * w_term * math.sin(heading_rad)) if glide_heading_deg is not None else 0.0
    glide_vy = (glide_ratio * w_term * math.cos(heading_rad)) if glide_heading_deg is not None else 0.0

    for _ in range(num_depth_steps):
        # 1. Sample current at midpoint of depth step
        mid_z = cur_z + 0.5 * dz
        if current_field_fn is not None:
            u_curr, v_curr = current_field_fn(cur_x, cur_y, mid_z)
        else:
            u_curr, v_curr = 0.0, 0.0

        # 2. Advance horizontal coordinates
        cur_x += (u_curr + glide_vx) * dt
        cur_y += (v_curr + glide_vy) * dt
        cur_z += dz

        waypoints.append((cur_x, cur_y, cur_z))

    return cur_x, cur_y, total_descent_time_s, waypoints


def simulate_3d_ascent_backtrack(
    seafloor_x_m: float,
    seafloor_y_m: float,
    seafloor_depth_m: float,
    profile: VesselProfile,
    current_field_fn: Optional[Callable[[float, float, float], Tuple[float, float]]] = None,
    glide_heading_deg: Optional[float] = None,
    num_depth_steps: int = 50,
) -> Tuple[float, float, float]:
    """Reverse 3D descent from seafloor wreck to calculate Surface Point of Sink (SPS).

    Inverts the velocity vectors (-u_current, -v_current, -glide) as particle ascends from
    bottom to surface (z: seafloor_depth -> 0).

    Args:
        seafloor_x_m: Known seafloor Eastward coordinate in meters.
        seafloor_y_m: Known seafloor Northward coordinate in meters.
        seafloor_depth_m: Water depth at wreck site in meters.
        profile: VesselProfile for descent velocity and glide ratio.
        current_field_fn: Optional function (x_m, y_m, depth_m) -> (u_current, v_current).
        glide_heading_deg: Heading angle of original forward glide.
        num_depth_steps: Number of integration steps.

    Returns:
        Tuple of (sps_x_m, sps_y_m, ascent_duration_s).
    """
    if seafloor_depth_m <= 0:
        raise ValueError(f"seafloor_depth_m must be positive, got {seafloor_depth_m}")

    w_term = profile.terminal_descent_rate_m_s
    ascent_time_s = seafloor_depth_m / w_term

    dz = seafloor_depth_m / float(num_depth_steps)
    dt = dz / w_term

    cur_x = seafloor_x_m
    cur_y = seafloor_y_m
    cur_z = seafloor_depth_m

    glide_ratio = profile.glide_ratio
    heading_rad = (
        math.radians(glide_heading_deg % 360.0)
        if glide_heading_deg is not None
        else 0.0
    )
    glide_vx = (glide_ratio * w_term * math.sin(heading_rad)) if glide_heading_deg is not None else 0.0
    glide_vy = (glide_ratio * w_term * math.cos(heading_rad)) if glide_heading_deg is not None else 0.0

    for _ in range(num_depth_steps):
        mid_z = cur_z - 0.5 * dz
        if current_field_fn is not None:
            u_curr, v_curr = current_field_fn(cur_x, cur_y, mid_z)
        else:
            u_curr, v_curr = 0.0, 0.0

        # Inverse integration: subtract forward velocities
        cur_x -= (u_curr + glide_vx) * dt
        cur_y -= (v_curr + glide_vy) * dt
        cur_z -= dz

    return cur_x, cur_y, ascent_time_s
