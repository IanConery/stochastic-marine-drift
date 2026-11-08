"""Unit and integration tests for Spatio-Temporal Convoy Breakdown Graph Neural Network."""

import os
import pytest
import torch

from src.models.convoy_gnn import (
    ConvoyBreakdownGNN,
    ConvoyCohesionEvaluator,
    load_convoy_network,
    train_convoy_gnn,
)
from src.modules.extreme_event import TropicalCyclone


def test_load_convoy_network():
    """Verify loading historical Carrera de Indias navigation network into DiGraph and PyG Data."""
    graph, node_to_idx, edge_records, pyg_data = load_convoy_network(
        "data/networks/caribbean_convoy_routes.geojson"
    )

    assert len(graph.nodes) == 21
    assert len(edge_records) == 25
    assert len(node_to_idx) == 21

    # Verify key historical ports, roadsteads, and choke points
    expected_nodes = [
        "havana",
        "cartagena",
        "veracruz",
        "portobelo",
        "port_royal",
        "dry_tortugas",
        "isle_of_pines",
        "florida_straits_marquesas",
        "florida_straits_canaveral",
        "mona_passage",
    ]
    for nid in expected_nodes:
        assert nid in graph.nodes
        assert nid in node_to_idx

    # Verify PyTorch Geometric tensor properties
    assert pyg_data.x.shape == (21, 4)
    assert pyg_data.edge_index.shape == (2, 25)
    assert pyg_data.edge_attr.shape == (25, 4)


def test_convoy_gnn_forward_pass():
    """Verify ConvoyBreakdownGNN outputs valid probability range and bounded scatter vectors."""
    model = ConvoyBreakdownGNN(node_dim=6, edge_dim=5, hidden_dim=16)

    num_nodes = 14
    num_edges = 13
    x = torch.randn(num_nodes, 6)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_attr = torch.randn(4, 5)

    p_detach, scatter_vecs = model(x, edge_index, edge_attr)

    assert p_detach.shape == (4, 1)
    assert scatter_vecs.shape == (4, 2)

    # Probabilities strictly in [0, 1]
    assert (p_detach >= 0.0).all() and (p_detach <= 1.0).all()

    # Scatter speed must be bounded (<= 3.0 m/s from tanh scaling)
    speeds = torch.norm(scatter_vecs, dim=-1)
    assert (speeds <= 3.01).all()


def test_dynamic_features_construction():
    """Verify dynamic node and edge features scale properly under hurricane presence."""
    evaluator = ConvoyCohesionEvaluator()

    # 1. Calm baseline
    x_calm, e_calm = evaluator.build_dynamic_features(cyclone=None)
    assert x_calm.shape == (21, 6)
    assert e_calm.shape == (25, 5)
    # Storm intensity in edge features should be zero
    assert torch.all(e_calm[:, 4] == 0.0)

    # 2. Hurricane over Florida Straits (centered 55km south of Marquesas in eyewall)
    cyclone = TropicalCyclone(
        storm_name="Atocha_Hurricane_1622",
        year=1622,
        eye_lat=24.0,
        eye_lon=-82.0,
        v_max_m_s=55.0,
        p_central_hpa=940.0,
        r_max_km=45.0,
    )
    x_storm, e_storm = evaluator.build_dynamic_features(cyclone=cyclone)

    # Nodes near Marquesas/Havana should register high wind and non-zero storm intensity
    marquesas_idx = evaluator.node_to_idx["florida_straits_marquesas"]
    assert x_storm[marquesas_idx, 2].item() > x_calm[marquesas_idx, 2].item()  # Higher wind
    assert torch.any(e_storm[:, 4] > 0.0)  # Storm intensity non-zero on nearby edges


def test_convoy_transit_evaluation_storm_vs_calm():
    """Verify storm conditions significantly elevate detachment probability and straggler count."""
    evaluator = ConvoyCohesionEvaluator()
    route = ["havana_marquesas", "marquesas_canaveral"]

    # 1. Calm transit
    res_calm = evaluator.evaluate_fleet_convoy(
        route_corridor_ids=route,
        fleet_size=20,
        cyclone=None,
    )

    assert res_calm["fleet_size"] == 20
    assert len(res_calm["edge_evaluations"]) == 2

    # 2. Severe hurricane hitting Marquesas corridor (centered 55km south in eyewall)
    hurricane = TropicalCyclone(
        storm_name="Atocha_1622",
        year=1622,
        eye_lat=24.0,
        eye_lon=-82.0,
        v_max_m_s=55.0,
        p_central_hpa=940.0,
        r_max_km=45.0,
    )
    res_storm = evaluator.evaluate_fleet_convoy(
        route_corridor_ids=route,
        fleet_size=20,
        cyclone=hurricane,
    )

    assert res_storm["total_expected_stragglers"] >= res_calm["total_expected_stragglers"]
    # Cohesion retention under hurricane must drop compared to calm conditions
    assert res_storm["cohesion_retention_rate"] <= res_calm["cohesion_retention_rate"]


def test_geojson_export():
    """Verify GeoJSON export includes LineStrings for convoy corridors and Points for stragglers."""
    evaluator = ConvoyCohesionEvaluator()
    route = ["havana_marquesas", "marquesas_canaveral"]
    eval_res = evaluator.evaluate_fleet_convoy(route_corridor_ids=route, fleet_size=15)

    geojson = evaluator.to_geojson(eval_res)
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 4  # 2 LineString edges + 2 Point straggler projections

    feature_types = {f["geometry"]["type"] for f in geojson["features"]}
    assert "LineString" in feature_types
    assert "Point" in feature_types


def test_train_convoy_gnn():
    """Verify training ConvoyBreakdownGNN converges and saves model weights."""
    evaluator = ConvoyCohesionEvaluator()
    losses = train_convoy_gnn(evaluator, epochs=15, lr=0.02)

    assert len(losses) == 15
    assert losses[-1] < losses[0]
    assert os.path.exists("data/processed/convoy_gnn.pt")
