"""Unit and integration tests for Physics-Informed Neural ODE (PINN-ODE) inverse dynamics."""

import torch
import pytest

from src.models.neural_ode import (
    ContinuousODESolver,
    PhysicsInformedNeuralODE,
    ReverseDriftVectorField,
)
from src.profiles.presets import SPANISH_GALLEON_1620
from src.utils.geo import knots_to_m_s


def test_reverse_vector_field_shape_and_speed_bound():
    """Verify output tensor shapes and strict velocity soft-clipping."""
    max_knots = 6.0
    max_ms = knots_to_m_s(max_knots)

    vf = ReverseDriftVectorField(
        state_dim=2,
        env_dim=6,
        hidden_dim=32,
        max_speed_ms=max_ms,
    )

    batch_size = 8
    x = torch.randn(batch_size, 2) * 1000.0  # Large positions
    t = torch.zeros(batch_size, 1)
    u_env = torch.randn(batch_size, 6) * 10.0  # High environmental forcing

    v = vf(x, t, u_env)
    assert v.shape == (batch_size, 2)

    # Compute speeds and verify strict adherence to max_speed_ms bound
    speeds = torch.norm(v, dim=-1)
    for s in speeds:
        assert s.item() <= (max_ms + 1e-4)


def test_continuous_ode_solver_rk4():
    """Verify ContinuousODESolver integrates trajectories smoothly over timestamps."""
    vf = ReverseDriftVectorField(hidden_dim=16)
    x0 = torch.tensor([[100.0, -200.0]], dtype=torch.float32)
    t_span = torch.tensor([0.0, 100.0, 200.0, 300.0], dtype=torch.float32)
    u_env = torch.zeros(1, 6, dtype=torch.float32)

    traj = ContinuousODESolver.integrate(vf, x0, t_span, u_env, method="rk4")

    # Shape should be (len(t_span), batch_size, 2)
    assert traj.shape == (4, 1, 2)
    # First waypoint matches initial state x0 exactly
    assert torch.allclose(traj[0], x0)


def test_navier_stokes_incompressibility_loss():
    """Verify spatial divergence du/dx + dv/dy computation via autograd."""
    pinn = PhysicsInformedNeuralODE(profile=SPANISH_GALLEON_1620, hidden_dim=16)

    x = torch.randn(4, 2)
    t = torch.zeros(4, 1)
    u_env = torch.randn(4, 6)

    ns_loss = pinn.compute_ns_divergence_loss(x, t, u_env)
    assert isinstance(ns_loss, torch.Tensor)
    assert ns_loss.ndim == 0  # Scalar
    assert ns_loss.item() >= 0.0


def test_speed_penalty_calculation():
    """Verify speed penalty is zero when below threshold and positive when above."""
    pinn = PhysicsInformedNeuralODE(max_velocity_knots=6.0)
    max_ms = pinn.max_speed_ms

    # Below threshold velocities
    v_sub = torch.tensor([[1.0, 1.0]], dtype=torch.float32)  # norm ~ 1.41 < 3.08
    assert pinn.compute_speed_penalty(v_sub).item() == 0.0

    # Above threshold velocities
    v_super = torch.tensor([[5.0, 5.0]], dtype=torch.float32)  # norm ~ 7.07 > 3.08
    penalty = pinn.compute_speed_penalty(v_super)
    assert penalty.item() > 0.0


def test_pinn_loss_decomposition():
    """Verify evaluate_losses computes data, Navier-Stokes, and speed penalty components."""
    pinn = PhysicsInformedNeuralODE(hidden_dim=16)

    t_span = torch.linspace(0.0, 600.0, 4)
    x0 = torch.zeros(2, 2)
    u_env = torch.ones(2, 6)

    pred_traj = pinn.forward(x0, t_span, u_env)
    target_traj = pred_traj + 5.0  # Synthetic offset

    losses = pinn.evaluate_losses(pred_traj, target_traj, t_span, u_env)

    assert "loss_data" in losses
    assert "loss_ns" in losses
    assert "loss_speed" in losses
    assert "loss_total" in losses

    assert losses["loss_data"].item() > 0.0
    assert losses["loss_total"].item() > 0.0


def test_pinn_fit_convergence():
    """Verify gradient descent fitting decreases total PINN loss."""
    torch.manual_seed(42)
    pinn = PhysicsInformedNeuralODE(hidden_dim=32)

    t_span = torch.linspace(0.0, 300.0, 5)
    x0 = torch.zeros(1, 2)
    u_env = torch.tensor([[10.0, -5.0, 0.2, -0.1, 0.05, -0.02]])

    # Target trajectory is a steady reverse drift
    target_traj = torch.stack([
        x0 - t * 1.5 for t in t_span
    ], dim=0)

    history = pinn.fit(
        x0=x0,
        target_trajectory=target_traj,
        t_span=t_span,
        u_env=u_env,
        epochs=15,
        lr=0.02,
    )

    assert len(history["loss_total"]) == 15
    # Total loss should decrease from epoch 0 to epoch 14
    assert history["loss_total"][-1] < history["loss_total"][0]


def test_predict_continuous_corridor_geojson():
    """Verify continuous corridor prediction outputs valid GeoJSON LineString."""
    pinn = PhysicsInformedNeuralODE(profile=SPANISH_GALLEON_1620, hidden_dim=16)

    corridor = pinn.predict_continuous_corridor(
        ref_lat=24.50,
        ref_lon=-82.20,
        duration_hours=2.0,
        u_env_dict={"u_wind": 12.0, "v_wind": -6.0, "u_curr": 0.25, "v_curr": -0.10},
        num_steps=10,
    )

    assert "waypoints" in corridor
    assert len(corridor["waypoints"]) == 10
    assert "origin_estimate" in corridor

    geojson = corridor["geojson"]
    assert geojson["type"] == "Feature"
    assert geojson["geometry"]["type"] == "LineString"
    assert len(geojson["geometry"]["coordinates"]) == 10
