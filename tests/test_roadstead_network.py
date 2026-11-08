"""Unit tests for colonial roadsteads, submerged ports, and convoy routing graph expansion."""

from pathlib import Path
import pytest

from src.models.convoy_gnn import load_convoy_network
from src.modules.roadstead_network import (
    CorridorEdge,
    RoadsteadNetworkBuilder,
    RoadsteadNode,
)


@pytest.fixture
def base_network_path() -> Path:
    return Path("data/networks/caribbean_convoy_routes.geojson")


def test_roadstead_node_serialization():
    """Verify that RoadsteadNode serializes into valid GeoJSON Point feature."""
    node = RoadsteadNode(
        node_id="port_royal",
        name="Port Royal (Submerged Harbor)",
        node_type="submerged_port",
        latitude=17.937,
        longitude=-76.841,
        significance="Submerged 1692",
        holding_quality="submerged",
        shelter_bearing_deg=180.0,
        max_vessel_capacity=100,
    )
    feat = node.to_feature()
    assert feat["type"] == "Feature"
    assert feat["geometry"]["type"] == "Point"
    assert feat["geometry"]["coordinates"] == [-76.841, 17.937]
    assert feat["properties"]["node_id"] == "port_royal"
    assert feat["properties"]["type"] == "submerged_port"


def test_corridor_edge_serialization():
    """Verify that CorridorEdge serializes into valid GeoJSON LineString feature."""
    edge = CorridorEdge(
        edge_id="cartagena_port_royal",
        source="cartagena",
        target="port_royal",
        name="Cartagena to Port Royal",
        route_type="diversion",
        distance_nmi=450.0,
        hazard_level=0.45,
        prevailing_current_knots=1.2,
        prevailing_wind_heading_deg=280.0,
        coordinates=[[-75.55, 10.42], [-76.841, 17.937]],
    )
    feat = edge.to_feature()
    assert feat["type"] == "Feature"
    assert feat["geometry"]["type"] == "LineString"
    assert feat["geometry"]["coordinates"] == [[-75.55, 10.42], [-76.841, 17.937]]
    assert feat["properties"]["route_type"] == "diversion"
    assert feat["properties"]["distance_nmi"] == 450.0


def test_build_expanded_network(base_network_path: Path):
    """Verify building expanded network contains all roadsteads, submerged ports, and corridors."""
    builder = RoadsteadNetworkBuilder()
    expanded = builder.build_expanded_network(base_network_path)

    features = expanded["features"]
    nodes = [f for f in features if f["geometry"]["type"] == "Point"]
    edges = [f for f in features if f["geometry"]["type"] == "LineString"]

    # 14 base nodes + 7 expansion nodes = 21 nodes
    assert len(nodes) == 21
    # 13 base edges + 12 expansion edges = 25 edges
    assert len(edges) == 25

    node_ids = {n["properties"]["node_id"] for n in nodes}
    expected_expansion_nodes = [
        "port_royal",
        "nombre_de_dios",
        "dry_tortugas",
        "isle_of_pines",
        "roatan",
        "virgin_gorda",
        "saint_eustatius",
    ]
    for exp_id in expected_expansion_nodes:
        assert exp_id in node_ids

    # Verify edge connectivity: all edge endpoints must exist in node_ids
    for e in edges:
        src = e["properties"]["source"]
        tgt = e["properties"]["target"]
        assert src in node_ids, f"Edge source {src} not found in node_ids"
        assert tgt in node_ids, f"Edge target {tgt} not found in node_ids"


def test_gnn_compatibility_with_expanded_network(tmp_path: Path, base_network_path: Path):
    """Verify that PyG and NetworkX load_convoy_network flawlessly ingests the expanded network."""
    builder = RoadsteadNetworkBuilder()
    expanded = builder.build_expanded_network(base_network_path)

    temp_geojson = tmp_path / "test_expanded_convoy_routes.geojson"
    builder.export_geojson(expanded, temp_geojson)

    graph, node_to_idx, edge_records, pyg_data = load_convoy_network(str(temp_geojson))

    assert len(graph.nodes) == 21
    assert len(edge_records) == 25
    assert len(node_to_idx) == 21

    # Verify PyG tensor dimensions
    assert pyg_data.x.shape == (21, 4)
    assert pyg_data.edge_index.shape == (2, 25)
    assert pyg_data.edge_attr.shape == (25, 4)
    assert pyg_data.node_coords.shape == (21, 2)
