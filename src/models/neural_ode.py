"""Continuous Physics-Informed Neural ODE (PINN-ODE) for reverse trajectory reconstruction."""

import json
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from src.profiles.presets import SPANISH_GALLEON_1620
from src.profiles.vessel_class import VesselProfile
from src.utils.geo import (
    METERS_PER_DEGREE_LAT,
    knots_to_m_s,
    lat_lon_to_meters_offsets,
    meters_to_lat_lon_offsets,
)


class ReverseDriftVectorField(nn.Module):
    """Neural network parameterizing the continuous reverse drift velocity field dx/dt = f_theta(x, t, u)."""

    def __init__(
        self,
        state_dim: int = 2,
        env_dim: int = 6,
        hidden_dim: int = 64,
        max_speed_ms: float = 3.087,
    ) -> None:
        """Initialize the reverse vector field module.

        Args:
            state_dim: Spatial dimension (2 for [x, y] in meters).
            env_dim: Dimension of environmental forcing inputs [u_wind, v_wind, u_curr, v_curr, u_stk, v_stk].
            hidden_dim: Number of hidden units in MLP layers.
            max_speed_ms: Vessel physical speed limit in m/s (default: ~6.0 knots = 3.087 m/s).
        """
        super().__init__()
        self.state_dim = state_dim
        self.env_dim = env_dim
        self.max_speed_ms = max_speed_ms

        # Total input: [x, y, t] + env_features
        input_dim = state_dim + 1 + env_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Initialize final layer with small weights for smooth near-linear starting behavior
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        u_env: torch.Tensor,
    ) -> torch.Tensor:
        """Compute instantaneous reverse velocity vector dx/dt.

        Args:
            x: Spatial coordinates tensor of shape (batch_size, 2) in meters.
            t: Continuous time tensor of shape (batch_size, 1) in seconds.
            u_env: Environmental forcing vector of shape (batch_size, env_dim).

        Returns:
            Reverse velocity tensor dx/dt of shape (batch_size, 2) in m/s.
        """
        inp = torch.cat([x, t, u_env], dim=-1)
        v_raw = self.net(inp)

        # Baseline physical reverse velocity: inverting environmental currents
        # u_env[:, 2:4] is [u_curr, v_curr], u_env[:, 4:6] is [u_stokes, v_stokes]
        if u_env.shape[-1] >= 6:
            u_baseline = -(u_env[:, 2:4] + u_env[:, 4:6])
        elif u_env.shape[-1] >= 4:
            u_baseline = -u_env[:, 2:4]
        else:
            u_baseline = torch.zeros_like(x)

        v_eff = v_raw + u_baseline

        # Smooth soft-clipping to prevent runaway velocities
        speed = torch.norm(v_eff, dim=-1, keepdim=True).clamp(min=1e-6)
        scale = torch.tanh(speed / self.max_speed_ms) * (self.max_speed_ms / speed)
        return v_eff * scale


class ContinuousODESolver:
    """Numerical integrator for continuous-time ODE trajectories in PyTorch."""

    @staticmethod
    def rk4_step(
        func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        u_env: torch.Tensor,
    ) -> torch.Tensor:
        """Perform a single 4th-order Runge-Kutta integration step."""
        dt_tensor = torch.tensor(dt, dtype=x.dtype, device=x.device)
        half_dt = dt_tensor * 0.5

        k1 = func(x, t, u_env)
        k2 = func(x + half_dt * k1, t + half_dt, u_env)
        k3 = func(x + half_dt * k2, t + half_dt, u_env)
        k4 = func(x + dt_tensor * k3, t + dt_tensor, u_env)

        return x + (dt_tensor / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    @staticmethod
    def integrate(
        func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        x0: torch.Tensor,
        t_span: torch.Tensor,
        u_env: torch.Tensor,
        method: str = "rk4",
    ) -> torch.Tensor:
        """Integrate continuous ODE trajectory over a sequence of timestamps.

        Args:
            func: ODE dynamics function f(x, t, u).
            x0: Initial state tensor of shape (batch_size, state_dim).
            t_span: 1D tensor of timestamps [t_0, t_1, ..., t_N].
            u_env: Environmental conditions tensor of shape (batch_size, env_dim).
            method: Integration method ('rk4' or 'euler').

        Returns:
            Trajectory tensor of shape (len(t_span), batch_size, state_dim).
        """
        trajectory = [x0]
        cur_x = x0

        for i in range(len(t_span) - 1):
            t_cur = t_span[i].unsqueeze(0).expand(cur_x.shape[0], 1)
            dt = float(t_span[i + 1] - t_span[i])

            if method == "rk4":
                cur_x = ContinuousODESolver.rk4_step(func, cur_x, t_cur, dt, u_env)
            else:
                # Standard Euler step
                v = func(cur_x, t_cur, u_env)
                cur_x = cur_x + v * dt

            trajectory.append(cur_x)

        return torch.stack(trajectory, dim=0)


class PhysicsInformedNeuralODE(nn.Module):
    """PINN-ODE combining continuous adjoint dynamics with Navier-Stokes and velocity regularization."""

    def __init__(
        self,
        profile: VesselProfile = SPANISH_GALLEON_1620,
        max_velocity_knots: float = 6.0,
        lambda_ns: float = 0.1,
        lambda_speed: float = 1.0,
        hidden_dim: int = 64,
    ) -> None:
        r"""Initialize the continuous PINN-ODE module.

        Args:
            profile: Vessel profile defining naval architecture constraints.
            max_velocity_knots: Speed ceiling ($V_{max} \le 6\text{ knots}$) in knots.
            lambda_ns: Weight for Navier-Stokes incompressibility penalty (div V = 0).
            lambda_speed: Weight for velocity ceiling penalty.
            hidden_dim: Width of MLP layers.
        """
        super().__init__()
        self.profile = profile
        self.max_speed_ms = knots_to_m_s(max_velocity_knots)
        self.lambda_ns = lambda_ns
        self.lambda_speed = lambda_speed

        self.vector_field = ReverseDriftVectorField(
            state_dim=2,
            env_dim=6,
            hidden_dim=hidden_dim,
            max_speed_ms=self.max_speed_ms,
        )
        self.solver = ContinuousODESolver()

    def forward(
        self,
        x0: torch.Tensor,
        t_span: torch.Tensor,
        u_env: torch.Tensor,
        method: str = "rk4",
    ) -> torch.Tensor:
        """Integrate continuous reverse trajectory from initial position x0 over t_span.

        Args:
            x0: Initial coordinate tensor (batch_size, 2) in meters.
            t_span: Timestamps tensor of shape (num_steps,).
            u_env: Environmental conditions (batch_size, 6).
            method: 'rk4' or 'euler'.

        Returns:
            Tensor of shape (num_steps, batch_size, 2) containing continuous positions.
        """
        return self.solver.integrate(self.vector_field, x0, t_span, u_env, method=method)

    def compute_ns_divergence_loss(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        u_env: torch.Tensor,
    ) -> torch.Tensor:
        """Compute fluid continuity / incompressibility penalty: div(V) = du/dx + dv/dy = 0.

        Args:
            x: Position tensor requiring gradients.
            t: Time tensor.
            u_env: Environmental features tensor.

        Returns:
            Scalar tensor representing mean squared divergence.
        """
        x_in = x.clone().detach().requires_grad_(True)
        v = self.vector_field(x_in, t, u_env)

        u_vel = v[:, 0]
        v_vel = v[:, 1]

        grad_u = torch.autograd.grad(
            u_vel.sum(), x_in, create_graph=True, retain_graph=True
        )[0]
        grad_v = torch.autograd.grad(
            v_vel.sum(), x_in, create_graph=True, retain_graph=True
        )[0]

        div = grad_u[:, 0] + grad_v[:, 1]
        return torch.mean(div**2)

    def compute_speed_penalty(
        self,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Compute velocity constraint penalty: max(0, ||V|| - V_max)^2."""
        speeds = torch.norm(v, dim=-1)
        excess = torch.relu(speeds - self.max_speed_ms)
        return torch.mean(excess**2)

    def evaluate_losses(
        self,
        pred_trajectory: torch.Tensor,
        target_trajectory: Optional[torch.Tensor],
        t_span: torch.Tensor,
        u_env: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute full PINN loss decomposition: Data + Navier-Stokes + Speed Limit.

        Args:
            pred_trajectory: Predicted trajectory tensor (num_steps, batch_size, 2).
            target_trajectory: Optional ground-truth anchor trajectory.
            t_span: Timestamps tensor (num_steps,).
            u_env: Environmental forcing tensor (batch_size, 6).

        Returns:
            Dict containing 'loss_data', 'loss_ns', 'loss_speed', and 'loss_total'.
        """
        # 1. Data loss
        if target_trajectory is not None:
            loss_data = nn.functional.mse_loss(pred_trajectory, target_trajectory)
        else:
            loss_data = torch.tensor(0.0, device=pred_trajectory.device)

        # 2. Navier-Stokes divergence loss sampled across trajectory midpoints
        mid_idx = len(t_span) // 2
        x_sample = pred_trajectory[mid_idx]
        t_sample = t_span[mid_idx].unsqueeze(0).expand(x_sample.shape[0], 1)
        loss_ns = self.compute_ns_divergence_loss(x_sample, t_sample, u_env)

        # 3. Speed bounding loss
        velocities = self.vector_field(x_sample, t_sample, u_env)
        loss_speed = self.compute_speed_penalty(velocities)

        loss_total = loss_data + self.lambda_ns * loss_ns + self.lambda_speed * loss_speed

        return {
            "loss_data": loss_data,
            "loss_ns": loss_ns,
            "loss_speed": loss_speed,
            "loss_total": loss_total,
        }

    def fit(
        self,
        x0: torch.Tensor,
        target_trajectory: torch.Tensor,
        t_span: torch.Tensor,
        u_env: torch.Tensor,
        epochs: int = 30,
        lr: float = 0.005,
    ) -> Dict[str, List[float]]:
        """Fit neural ODE parameters to observed trajectory while minimizing physics penalties.

        Args:
            x0: Initial coordinate tensor (batch_size, 2).
            target_trajectory: Target trajectory (num_steps, batch_size, 2).
            t_span: Timestamps (num_steps,).
            u_env: Environmental conditions (batch_size, 6).
            epochs: Number of gradient descent optimization steps.
            lr: Learning rate for Adam optimizer.

        Returns:
            Dict of loss convergence histories.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history: Dict[str, List[float]] = {
            "loss_total": [],
            "loss_data": [],
            "loss_ns": [],
            "loss_speed": [],
        }

        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = self.forward(x0, t_span, u_env, method="rk4")
            losses = self.evaluate_losses(pred, target_trajectory, t_span, u_env)
            losses["loss_total"].backward()
            optimizer.step()

            for k in history:
                history[k].append(float(losses[k].item()))

        return history

    def predict_continuous_corridor(
        self,
        ref_lat: float,
        ref_lon: float,
        duration_hours: float,
        u_env_dict: Dict[str, float],
        num_steps: int = 50,
    ) -> Dict[str, Any]:
        """Generate smooth continuous reverse-time trajectory corridor and GeoJSON representation.

        Args:
            ref_lat: Base latitude (e.g. Surface Point of Sink).
            ref_lon: Base longitude.
            duration_hours: Reverse time duration in hours.
            u_env_dict: Dictionary with 'u_wind', 'v_wind', 'u_curr', 'v_curr', 'u_stk', 'v_stk'.
            num_steps: Number of continuous integration points.

        Returns:
            Dictionary containing waypoints, corridor bounds, and GeoJSON track.
        """
        self.eval()
        t_total_s = duration_hours * 3600.0
        t_span = torch.linspace(0.0, t_total_s, num_steps)

        # Initial point is (0, 0) relative to ref_lat, ref_lon
        x0 = torch.zeros((1, 2), dtype=torch.float32)

        u_env_tensor = torch.tensor(
            [[
                u_env_dict.get("u_wind", 10.0),
                u_env_dict.get("v_wind", -5.0),
                u_env_dict.get("u_curr", 0.20),
                u_env_dict.get("v_curr", -0.10),
                u_env_dict.get("u_stk", 0.05),
                u_env_dict.get("v_stk", -0.02),
            ]],
            dtype=torch.float32,
        )

        with torch.no_grad():
            trajectory = self.forward(x0, t_span, u_env_tensor, method="rk4")  # (num_steps, 1, 2)

        coords_lat_lon: List[Tuple[float, float, float]] = []
        geojson_coords: List[List[float]] = []

        for idx in range(num_steps):
            t_s = float(t_span[idx].item())
            dx_m = float(trajectory[idx, 0, 0].item())
            dy_m = float(trajectory[idx, 0, 1].item())

            dlat, dlon = meters_to_lat_lon_offsets(dx_m, dy_m, ref_lat)
            lat = ref_lat + dlat
            lon = ref_lon + dlon

            coords_lat_lon.append((t_s, round(lat, 6), round(lon, 6)))
            geojson_coords.append([round(lon, 6), round(lat, 6)])

        # Construct GeoJSON Feature
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": geojson_coords,
            },
            "properties": {
                "model": "PhysicsInformedNeuralODE",
                "duration_hours": duration_hours,
                "num_steps": num_steps,
                "vessel_class": self.profile.class_name,
                "max_speed_knots": round(self.max_speed_ms * 1.94384, 2),
            },
        }

        return {
            "timestamps_s": [c[0] for c in coords_lat_lon],
            "waypoints": coords_lat_lon,
            "origin_estimate": coords_lat_lon[-1],
            "geojson": feature,
        }
