"""Core simulation engines, forward drift models, inverse backtrackers, neural ODEs, survival models, convoy GNNs, post-depositional models, and validation benchmarks."""

from src.models.backtrack import (
    BacktrackParticle,
    BacktrackSimulationResult,
    IntersectionResult,
    ReverseLagrangianBacktracker,
)
from src.models.convoy_gnn import (
    ConvoyBreakdownGNN,
    ConvoyCohesionEvaluator,
    load_convoy_network,
    train_convoy_gnn,
)
from src.models.debris_field import (
    DebrisField,
    estimate_debris_field,
    generate_confidence_ellipse_points,
    meters_to_lat_lon_offsets,
)
from src.models.forward_drift import (
    DriftParticle,
    EnvironmentalConditions,
    ForwardDriftSimulator,
    ForwardSimulationResult,
    ParticleState,
)
from src.models.foundering import (
    BayesianPartialPooler,
    DynamicFounderingModel,
    FounderingObservation,
    SyntheticCohortGenerator,
    train_and_save_default_model,
)
from src.models.neural_ode import (
    ContinuousODESolver,
    PhysicsInformedNeuralODE,
    ReverseDriftVectorField,
)
from src.models.post_depositional import (
    PostDepositionalDispersalModel,
    PostDepositionalSiteState,
    StormImpactEvent,
    compute_bottom_orbital_velocity,
    solve_wave_wavenumber,
)
from src.models.validation import (
    HISTORICAL_BENCHMARKS,
    CaseBenchmarkResult,
    HistoricalBenchmarkRunner,
    HistoricalWreckTarget,
)

__all__ = [
    # Debris field
    "DebrisField",
    "estimate_debris_field",
    "generate_confidence_ellipse_points",
    "meters_to_lat_lon_offsets",
    # Forward drift
    "ParticleState",
    "DriftParticle",
    "EnvironmentalConditions",
    "ForwardSimulationResult",
    "ForwardDriftSimulator",
    # Backtrack
    "BacktrackParticle",
    "BacktrackSimulationResult",
    "IntersectionResult",
    "ReverseLagrangianBacktracker",
    # Neural ODE
    "ReverseDriftVectorField",
    "ContinuousODESolver",
    "PhysicsInformedNeuralODE",
    # Survival & Foundering
    "BayesianPartialPooler",
    "DynamicFounderingModel",
    "FounderingObservation",
    "SyntheticCohortGenerator",
    "train_and_save_default_model",
    # Convoy GNN
    "ConvoyBreakdownGNN",
    "ConvoyCohesionEvaluator",
    "load_convoy_network",
    "train_convoy_gnn",
    # Post-Depositional Dispersal
    "PostDepositionalDispersalModel",
    "PostDepositionalSiteState",
    "StormImpactEvent",
    "compute_bottom_orbital_velocity",
    "solve_wave_wavenumber",
    # Validation & Benchmarks
    "HISTORICAL_BENCHMARKS",
    "HistoricalWreckTarget",
    "HistoricalBenchmarkRunner",
    "CaseBenchmarkResult",
]
