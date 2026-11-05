"""Comprehensive unit tests for decoupled environmental physics modules."""

import math
import numpy as np
import pytest

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
from src.profiles.presets import SPANISH_GALLEON_1620, MODERN_CONTAINER_2015


# =========================================================================
# 1. Stokes Drift Tests
# =========================================================================

def test_stokes_speed_zero_wave():
    assert compute_surface_stokes_speed(0.0, 10.0) == 0.0


def test_stokes_invalid_inputs():
    with pytest.raises(ValueError, match="wave_height_m cannot be negative"):
        compute_surface_stokes_speed(-2.0, 8.0)
    with pytest.raises(ValueError, match="peak_period_s must be strictly positive"):
        compute_surface_stokes_speed(3.0, 0.0)


def test_stokes_drift_vector_directions():
    # Wave heading East (90 deg to) -> u > 0, v approx 0
    u_stokes, v_stokes = compute_stokes_drift_vector(
        wave_height_m=3.0, peak_period_s=8.0, wave_direction_deg=90.0, direction_convention="to"
    )
    assert u_stokes > 0.0
    assert abs(v_stokes) < 1e-6

    # Wave heading North (0 deg to) -> u approx 0, v > 0
    u_n, v_n = compute_stokes_drift_vector(
        wave_height_m=3.0, peak_period_s=8.0, wave_direction_deg=0.0, direction_convention="to"
    )
    assert abs(u_n) < 1e-6
    assert v_n > 0.0

    # 'from' convention reverses direction
    u_from, v_from = compute_stokes_drift_vector(
        wave_height_m=3.0, peak_period_s=8.0, wave_direction_deg=0.0, direction_convention="from"
    )
    assert abs(u_from) < 1e-6
    assert v_from < 0.0


def test_stokes_profile_depth_decay():
    surface = compute_stokes_profile(wave_height_m=4.0, peak_period_s=9.0, depth_m=0.0)
    deep_10m = compute_stokes_profile(wave_height_m=4.0, peak_period_s=9.0, depth_m=10.0)
    deep_50m = compute_stokes_profile(wave_height_m=4.0, peak_period_s=9.0, depth_m=50.0)

    assert surface > deep_10m > deep_50m > 0.0


# =========================================================================
# 2. Wind & Turbulence Tests
# =========================================================================

def test_wind_speed_direction_roundtrip():
    orig_u, orig_v = 8.5, -4.2
    speed, heading = wind_speed_and_direction(orig_u, orig_v)
    rec_u, rec_v = wind_components_from_speed_dir(speed, heading)

    assert math.isclose(orig_u, rec_u, rel_tol=1e-5, abs_tol=1e-5)
    assert math.isclose(orig_v, rec_v, rel_tol=1e-5, abs_tol=1e-5)


def test_weibull_turbulence_expectation():
    mean_speed = 12.0
    rng = np.random.default_rng(42)
    samples = [
        sample_weibull_turbulence(mean_speed, weibull_shape=2.0, turbulence_intensity=0.2, rng=rng)
        for _ in range(2000)
    ]
    empirical_mean = float(np.mean(samples))
    # Expectation should be close to mean_speed within 2%
    assert math.isclose(empirical_mean, mean_speed, rel_tol=0.02)


def test_land_sea_breeze_diurnal_phases():
    # Coast facing North (0 deg onshore)
    # At 15:00 (peak afternoon heating) -> onshore breeze (v > 0)
    u_day, v_day = compute_land_sea_breeze_perturbation(
        dist_to_coast_km=5.0, coast_onshore_heading_deg=0.0, local_solar_hour=15.0
    )
    assert v_day > 1.0

    # At 03:00 (peak nighttime cooling) -> offshore catabatic breeze (v < 0)
    u_night, v_night = compute_land_sea_breeze_perturbation(
        dist_to_coast_km=5.0, coast_onshore_heading_deg=0.0, local_solar_hour=3.0
    )
    assert v_night < -1.0

    # Spatial decay: 80 km offshore should be much weaker than 5 km offshore
    _, v_far = compute_land_sea_breeze_perturbation(
        dist_to_coast_km=80.0, coast_onshore_heading_deg=0.0, local_solar_hour=15.0
    )
    assert abs(v_far) < abs(v_day) * 0.15


def test_perturb_wind_vector_output():
    rng = np.random.default_rng(123)
    u_pert, v_pert = perturb_wind_vector(
        u10=10.0, v10=5.0, dist_to_coast_km=10.0, coast_onshore_heading_deg=90.0, local_solar_hour=14.0, rng=rng
    )
    assert isinstance(u_pert, float)
    assert isinstance(v_pert, float)
    assert not math.isnan(u_pert) and not math.isnan(v_pert)


# =========================================================================
# 3. Current & Ekman Shear Tests
# =========================================================================

def test_coriolis_parameter():
    assert coriolis_parameter(0.0) == 0.0
    f_caribbean = coriolis_parameter(20.0)
    assert f_caribbean > 0.0
    assert coriolis_parameter(-20.0) == -f_caribbean


def test_ekman_deflection_northern_hemisphere():
    # Northern Hemisphere Caribbean (lat 22N), wind blowing directly North (u=0, v=15 m/s)
    # Ekman deflection must be 45 deg to the RIGHT of the wind (Eastward, u > 0)
    u_ek, v_ek = compute_ekman_surface_velocity(u10=0.0, v10=15.0, latitude_deg=22.0)
    assert u_ek > 0.0
    assert v_ek > 0.0
    # At 45 deg deflection, u_ek and v_ek should be approximately equal
    assert math.isclose(u_ek, v_ek, rel_tol=1e-4)


def test_interpolate_current_with_depth():
    u_surf, v_surf = 0.5, 0.2
    u_deep, v_deep = 0.1, -0.05
    u_top, v_top = interpolate_current_with_depth(
        depth_m=0.0, u_surface=u_surf, v_surface=v_surf, u_deep=u_deep, v_deep=v_deep
    )
    assert (u_top, v_top) == (u_surf, v_surf)

    u_bot, v_bot = interpolate_current_with_depth(
        depth_m=100.0, u_surface=u_surf, v_surface=v_surf, u_deep=u_deep, v_deep=v_deep, deep_reference_depth_m=100.0
    )
    assert math.isclose(u_bot, u_deep, abs_tol=1e-5)
    assert math.isclose(v_bot, v_deep, abs_tol=1e-5)


# =========================================================================
# 4. Extreme Events & Holland Vortex Tests
# =========================================================================

def test_haversine_distance():
    # Miami (25.7617 N, 80.1918 W) to Havana (23.1136 N, 82.3666 W) ~ 368 km
    dist_km, bearing = haversine_distance_and_bearing(25.7617, -80.1918, 23.1136, -82.3666)
    assert 350.0 < dist_km < 390.0
    assert 190.0 < bearing < 220.0


def test_holland_vortex_evaluation():
    # Category 4 Hurricane Joaquin-like storm
    storm = TropicalCyclone(
        storm_name="Joaquin",
        year=2015,
        eye_lat=23.0,
        eye_lon=-74.0,
        v_max_m_s=60.0,
        p_central_hpa=935.0,
        p_ambient_hpa=1013.25,
        r_max_km=30.0,
    )

    # 1. At eye center (r < 1 km), rotational wind is minimal and pressure is central
    u_eye, v_eye, p_eye = storm.evaluate_at_point(23.0, -74.0)
    assert math.hypot(u_eye, v_eye) < 1.0
    assert math.isclose(p_eye, 935.0)

    # 2. Near eyewall (30 km away to the East at 23.0 N, -73.71 W)
    # Wind should be near hurricane force (> 40 m/s) and cyclonic (counter-clockwise in NH)
    u_wall, v_wall, p_wall = storm.evaluate_at_point(23.0, -73.71)
    wind_wall_speed = math.hypot(u_wall, v_wall)
    assert wind_wall_speed > 35.0
    assert 935.0 < p_wall < 1013.25


# =========================================================================
# 5. Deep Water 3D Descent & Ascent Tests
# =========================================================================

def test_terminal_descent_velocity():
    # Positively buoyant hull raises ValueError
    with pytest.raises(ValueError, match="positively buoyant"):
        compute_terminal_descent_velocity(
            displacement_volume_m3=100.0,
            vessel_mass_kg=50000.0,  # 50 metric tons in 100 m^3 water -> floats!
            projected_area_m2=30.0,
        )

    # Heavily ballasted sunken hull (e.g. 100 m^3 displacement, 250 metric tons mass)
    w_term = compute_terminal_descent_velocity(
        displacement_volume_m3=100.0,
        vessel_mass_kg=250000.0,
        projected_area_m2=40.0,
    )
    assert 1.0 < w_term < 10.0


def test_bi_directional_descent_ascent_symmetry():
    """Verify that 3D descent followed by reverse 3D ascent recovers original Surface Point of Sink."""
    galleon = SPANISH_GALLEON_1620
    sps_x = 1000.0
    sps_y = 5000.0
    depth_m = 600.0

    # Constant deep current field: Eastward 0.3 m/s, Northward -0.1 m/s
    def simple_current(x, y, z):
        return 0.3, -0.1

    # 1. Forward Descent
    bottom_x, bottom_y, descent_time, waypoints = simulate_3d_descent(
        sps_x_m=sps_x,
        sps_y_m=sps_y,
        seafloor_depth_m=depth_m,
        profile=galleon,
        current_field_fn=simple_current,
        glide_heading_deg=90.0,
        num_depth_steps=60,
    )

    assert bottom_x != sps_x
    assert bottom_y != sps_y
    assert len(waypoints) == 61

    # 2. Reverse Ascent Backtrack
    rec_sps_x, rec_sps_y, ascent_time = simulate_3d_ascent_backtrack(
        seafloor_x_m=bottom_x,
        seafloor_y_m=bottom_y,
        seafloor_depth_m=depth_m,
        profile=galleon,
        current_field_fn=simple_current,
        glide_heading_deg=90.0,
        num_depth_steps=60,
    )

    # Must recover the initial Surface Point of Sink with high numerical precision
    assert math.isclose(sps_x, rec_sps_x, rel_tol=1e-4, abs_tol=1e-2)
    assert math.isclose(sps_y, rec_sps_y, rel_tol=1e-4, abs_tol=1e-2)
    assert math.isclose(descent_time, ascent_time, rel_tol=1e-5)
