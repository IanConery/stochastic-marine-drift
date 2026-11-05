"""Spatio-Temporal Graph Neural Network (ST-GNN) for colonial Caribbean convoy breakdown prediction."""

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from src.modules.extreme_event import TropicalCyclone
from src.utils.geo import (
    METERS_PER_DEGREE_LAT,
    haversine_bearing_deg,
    haversine_distance_m,
    meters_to_lat_lon_offsets,
)


def load_convoy_network(
    geojson_path: str = "data/networks/caribbean_convoy_routes.geojson",
) -> Tuple[nx.DiGraph, Dict[str, int], List[Dict[str, Any]], Data]:
    """Load historical Caribbean convoy routes GeoJSON into NetworkX and PyTorch Geometric Data.

    Args:
        geojson_path: File path to caribbean_convoy_routes.geojson.

    Returns:
        Tuple of (DiGraph, node_to_idx, edge_records, pyg_data).
    """
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(f"Convoy route network file not found at: {geojson_path}")

    with open(geojson_path, "r", encoding="utf-8") as f:
        geo_dict = json.load(f)

    graph = nx.DiGraph()
    node_to_idx: Dict[str, int] = {}
    node_coords: List[List[float]] = []
    idx = 0

    # 1. Parse Node Points
    for feat in geo_dict["features"]:
        if feat["geometry"]["type"] == "Point":
            props = feat["properties"]
            nid = props["node_id"]
            lon, lat = feat["geometry"]["coordinates"]
            graph.add_node(
                nid,
                name=props.get("name", nid),
                node_type=props.get("type", "waypoint"),
                lat=lat,
                lon=lon,
            )
            node_to_idx[nid] = idx
            node_coords.append([lat, lon])
            idx += 1

    # 2. Parse Edge LineStrings
    edge_records: List[Dict[str, Any]] = []
    edge_indices: List[List[int]] = []
    edge_attributes: List[List[float]] = []

    for feat in geo_dict["features"]:
        if feat["geometry"]["type"] == "LineString":
            props = feat["properties"]
            src_id = props["source"]
            dst_id = props["target"]

            if src_id in node_to_idx and dst_id in node_to_idx:
                src_idx = node_to_idx[src_id]
                dst_idx = node_to_idx[dst_id]

                dist = float(props.get("distance_nmi", 100.0))
                hazard = float(props.get("hazard_level", 0.3))
                curr_spd = float(props.get("prevailing_current_knots", 1.0))
                wind_head = float(props.get("prevailing_wind_heading_deg", 270.0))

                graph.add_edge(
                    src_id,
                    dst_id,
                    edge_id=props.get("edge_id", f"{src_id}_{dst_id}"),
                    name=props.get("name", f"{src_id} to {dst_id}"),
                    distance_nmi=dist,
                    hazard_level=hazard,
                    current_knots=curr_spd,
                    wind_heading_deg=wind_head,
                    coordinates=feat["geometry"]["coordinates"],
                )

                edge_records.append({
                    "edge_id": props.get("edge_id", f"{src_id}_{dst_id}"),
                    "source": src_id,
                    "target": dst_id,
                    "src_idx": src_idx,
                    "dst_idx": dst_idx,
                    "distance_nmi": dist,
                    "hazard_level": hazard,
                    "current_knots": curr_spd,
                    "wind_heading_deg": wind_head,
                    "coordinates": feat["geometry"]["coordinates"],
                })

                edge_indices.append([src_idx, dst_idx])
                # Edge attribute vector: [normalized distance, hazard, current speed, wind heading / 360]
                edge_attributes.append([dist / 1000.0, hazard, curr_spd / 4.0, wind_head / 360.0])

    edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr_tensor = torch.tensor(edge_attributes, dtype=torch.float32)
    node_coords_tensor = torch.tensor(node_coords, dtype=torch.float32)

    # Base node feature: [lat / 90, lon / 180, is_port (1/0), is_choke_point (1/0)]
    node_features = []
    for nid, d in graph.nodes(data=True):
        is_port = 1.0 if d.get("node_type") == "port" else 0.0
        is_choke = 1.0 if d.get("node_type") == "choke_point" else 0.0
        node_features.append([d["lat"] / 90.0, d["lon"] / 180.0, is_port, is_choke])
    x_tensor = torch.tensor(node_features, dtype=torch.float32)

    pyg_data = Data(x=x_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)
    pyg_data.node_coords = node_coords_tensor

    return graph, node_to_idx, edge_records, pyg_data


class ConvoyBreakdownGNN(nn.Module):
    """Graph Attention Network predicting fleet detachment risk and straggler scatter vectors."""

    def __init__(
        self,
        node_dim: int = 6,
        edge_dim: int = 5,
        hidden_dim: int = 32,
    ) -> None:
        """Initialize GNN layers.

        Args:
            node_dim: Node feature dimension [lat, lon, wind_spd, wave_hs, dist_cyclone, fleet_density].
            edge_dim: Edge feature dimension [dist, baseline_hazard, current_spd, headwind_component, storm_intensity].
            hidden_dim: Hidden dimension size for graph attention layers.
        """
        super().__init__()
        self.conv1 = GATConv(node_dim, hidden_dim, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim, edge_dim=edge_dim)

        # Edge head: predicts probability P(detachment in [0, 1]) for each corridor
        self.detachment_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Scatter head: predicts straggler scatter drift offset [u_scatter, v_scatter] in m/s
        self.scatter_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Tanh(),  # Bounded scatter velocity (-3.0 to +3.0 m/s)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through graph attention network.

        Args:
            x: Node features tensor of shape (num_nodes, node_dim).
            edge_index: Graph connectivity tensor of shape (2, num_edges).
            edge_attr: Edge features tensor of shape (num_edges, edge_dim).

        Returns:
            Tuple of:
                - detachment_probabilities: Tensor of shape (num_edges, 1).
                - scatter_vectors: Tensor of shape (num_edges, 2) in m/s.
        """
        h = torch.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        h = self.conv2(h, edge_index, edge_attr=edge_attr)

        src_indices = edge_index[0]
        dst_indices = edge_index[1]

        # Concatenate source node, target node, and edge features
        edge_embeddings = torch.cat([h[src_indices], h[dst_indices], edge_attr], dim=-1)

        p_detachment = self.detachment_head(edge_embeddings)
        scatter_vecs = self.scatter_head(edge_embeddings) * 3.0  # Scale tanh to 3 m/s

        return p_detachment, scatter_vecs


class ConvoyCohesionEvaluator:
    """Evaluates fleet coordination, straggler separation, and scatter drift under storm forcing."""

    def __init__(
        self,
        geojson_path: str = "data/networks/caribbean_convoy_routes.geojson",
        model: Optional[ConvoyBreakdownGNN] = None,
        weights_path: Optional[str] = "data/processed/convoy_gnn.pt",
    ) -> None:
        """Initialize convoy evaluator with network graph and GNN model."""
        self.graph, self.node_to_idx, self.edge_records, self.base_data = load_convoy_network(geojson_path)
        self.idx_to_node = {idx: nid for nid, idx in self.node_to_idx.items()}

        if model is not None:
            self.model = model
        else:
            self.model = ConvoyBreakdownGNN(node_dim=6, edge_dim=5, hidden_dim=32)
            if weights_path and os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location="cpu")
                self.model.load_state_dict(state_dict)
            self.model.eval()

    def build_dynamic_features(
        self,
        cyclone: Optional[TropicalCyclone] = None,
        fleet_positions: Optional[Dict[str, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct dynamic node and edge feature tensors incorporating storm wind fields and fleet density."""
        num_nodes = len(self.node_to_idx)
        num_edges = len(self.edge_records)

        node_feats = np.zeros((num_nodes, 6), dtype=np.float32)
        edge_feats = np.zeros((num_edges, 5), dtype=np.float32)

        fleet_pos = fleet_positions or {}

        for nid, idx in self.node_to_idx.items():
            n_data = self.graph.nodes[nid]
            lat, lon = n_data["lat"], n_data["lon"]

            if cyclone is not None:
                dist_km = haversine_distance_m(lat, lon, cyclone.eye_lat, cyclone.eye_lon) / 1000.0
                u_w, v_w, _ = cyclone.evaluate_at_point(lat, lon)
                wind_spd = math.hypot(u_w, v_w)
                wave_hs = min(12.0, 0.22 * (wind_spd**1.2))
            else:
                dist_km = 1000.0
                wind_spd = 10.0
                wave_hs = 2.0

            vessel_count = float(fleet_pos.get(nid, 0))

            node_feats[idx] = [
                lat / 90.0,
                lon / 180.0,
                min(2.0, wind_spd / 25.0),
                min(2.0, wave_hs / 6.0),
                min(2.0, dist_km / 500.0),
                min(2.0, vessel_count / 10.0),
            ]

        for e_idx, e_rec in enumerate(self.edge_records):
            src_nid = e_rec["source"]
            dst_nid = e_rec["target"]
            src_data = self.graph.nodes[src_nid]
            dst_data = self.graph.nodes[dst_nid]

            bearing_deg = haversine_bearing_deg(
                src_data["lat"], src_data["lon"], dst_data["lat"], dst_data["lon"]
            )

            dist_nmi = e_rec["distance_nmi"]
            hazard = e_rec["hazard_level"]
            curr_spd = e_rec["current_knots"]

            if cyclone is not None:
                mid_lat = (src_data["lat"] + dst_data["lat"]) / 2.0
                mid_lon = (src_data["lon"] + dst_data["lon"]) / 2.0
                u_w, v_w, _ = cyclone.evaluate_at_point(mid_lat, mid_lon)
                wind_bearing = (math.degrees(math.atan2(u_w, v_w)) + 360.0) % 360.0
                # Headwind opposition angle: 1.0 when wind directly opposes route bearing
                angle_diff_rad = math.radians(abs((wind_bearing - bearing_deg + 180.0) % 360.0 - 180.0))
                headwind_component = math.cos(angle_diff_rad)  # Positive for opposing headwind
                dist_mid_cyclone = haversine_distance_m(mid_lat, mid_lon, cyclone.eye_lat, cyclone.eye_lon) / 1000.0
                storm_intensity = max(0.0, 1.0 - (dist_mid_cyclone / 400.0))
            else:
                headwind_component = 0.0
                storm_intensity = 0.0

            edge_feats[e_idx] = [
                dist_nmi / 1000.0,
                hazard,
                curr_spd / 4.0,
                headwind_component,
                storm_intensity,
            ]

        return torch.tensor(node_feats, dtype=torch.float32), torch.tensor(edge_feats, dtype=torch.float32)

    def evaluate_fleet_convoy(
        self,
        route_corridor_ids: List[str],
        fleet_size: int = 15,
        cyclone: Optional[TropicalCyclone] = None,
    ) -> Dict[str, Any]:
        """Evaluate fleet cohesion and predict straggler detachments along a convoy route.

        Args:
            route_corridor_ids: Ordered list of edge_ids traversed by the convoy.
            fleet_size: Total number of vessels sailing in convoy formation.
            cyclone: Optional TropicalCyclone active in the Caribbean basin.

        Returns:
            Dictionary containing edge evaluations, expected straggler counts, and scatter vectors.
        """
        self.model.eval()

        # Place fleet along the route corridors
        fleet_positions: Dict[str, int] = {}
        vessels_per_edge = max(1, fleet_size // max(1, len(route_corridor_ids)))
        for eid in route_corridor_ids:
            for rec in self.edge_records:
                if rec["edge_id"] == eid:
                    fleet_positions[rec["source"]] = fleet_positions.get(rec["source"], 0) + vessels_per_edge

        x, edge_attr = self.build_dynamic_features(cyclone=cyclone, fleet_positions=fleet_positions)

        with torch.no_grad():
            p_detach, scatter_vecs = self.model(x, self.base_data.edge_index, edge_attr)

        edge_evaluations: List[Dict[str, Any]] = []
        total_stragglers = 0.0

        for e_idx, rec in enumerate(self.edge_records):
            eid = rec["edge_id"]
            if eid in route_corridor_ids:
                prob = float(p_detach[e_idx, 0].item())
                scatter_u = float(scatter_vecs[e_idx, 0].item())
                scatter_v = float(scatter_vecs[e_idx, 1].item())

                scatter_spd_knots = math.hypot(scatter_u, scatter_v) * 1.94384
                scatter_bearing = (math.degrees(math.atan2(scatter_u, scatter_v)) + 360.0) % 360.0

                expected_stragglers = round(vessels_per_edge * prob, 2)
                total_stragglers += expected_stragglers

                edge_evaluations.append({
                    "edge_id": eid,
                    "name": rec.get("name", eid),
                    "source": rec["source"],
                    "target": rec["target"],
                    "detachment_probability": round(prob, 4),
                    "assigned_vessels": vessels_per_edge,
                    "expected_stragglers": expected_stragglers,
                    "scatter_vector_m_s": [round(scatter_u, 3), round(scatter_v, 3)],
                    "scatter_speed_knots": round(scatter_spd_knots, 2),
                    "scatter_bearing_deg": round(scatter_bearing, 1),
                })

        return {
            "fleet_size": fleet_size,
            "total_route_edges": len(route_corridor_ids),
            "total_expected_stragglers": round(total_stragglers, 1),
            "cohesion_retention_rate": round(max(0.0, 1.0 - (total_stragglers / max(1, fleet_size))), 3),
            "edge_evaluations": edge_evaluations,
            "high_risk_breakdown_edges": [
                e["edge_id"] for e in edge_evaluations if e["detachment_probability"] >= 0.50
            ],
        }

    def to_geojson(
        self, evaluation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Export convoy breakdown evaluation as a GeoJSON FeatureCollection."""
        features: List[Dict[str, Any]] = []
        eval_dict = {e["edge_id"]: e for e in evaluation_result["edge_evaluations"]}

        for rec in self.edge_records:
            eid = rec["edge_id"]
            if eid in eval_dict:
                ed = eval_dict[eid]
                coords = rec["coordinates"]

                # 1. Convoy Corridor LineString
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords,
                    },
                    "properties": {
                        "edge_id": eid,
                        "name": ed["name"],
                        "detachment_probability": ed["detachment_probability"],
                        "expected_stragglers": ed["expected_stragglers"],
                        "scatter_speed_knots": ed["scatter_speed_knots"],
                    },
                })

                # 2. Predicted Straggler Scatter Point (at corridor midpoint)
                mid_idx = len(coords) // 2
                base_lon, base_lat = coords[mid_idx]
                su, sv = ed["scatter_vector_m_s"]
                dlat, dlon = meters_to_lat_lon_offsets(su * 3600.0, sv * 3600.0, base_lat)

                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [round(base_lon + dlon, 6), round(base_lat + dlat, 6)],
                    },
                    "properties": {
                        "feature_type": "straggler_scatter_projection",
                        "parent_edge": eid,
                        "expected_stragglers": ed["expected_stragglers"],
                        "scatter_bearing_deg": ed["scatter_bearing_deg"],
                    },
                })

        return {"type": "FeatureCollection", "features": features}


def train_convoy_gnn(
    evaluator: ConvoyCohesionEvaluator,
    epochs: int = 40,
    lr: float = 0.01,
    save_path: str = "data/processed/convoy_gnn.pt",
) -> List[float]:
    """Train ConvoyBreakdownGNN on synthetic historical storm-crossing scenarios.

    Teaches the GNN the physical principle that edges with high opposing headwinds,
    severe hurricane storm intensity, and hazardous choke points have high detachment rates.
    """
    model = evaluator.model
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    loss_history: List[float] = []

    # Create synthetic storm scenarios
    cyclones = [
        # Severe hurricane passing over Marquesas / Florida Keys (like Atocha 1622)
        TropicalCyclone(
            storm_name="Atocha_1622",
            year=1622,
            eye_lat=24.5,
            eye_lon=-82.0,
            v_max_m_s=55.0,
            p_central_hpa=940.0,
            r_max_km=45.0,
        ),
        # Hurricane traversing northern Cuba and Canaveral (like 1715 Plate Fleet)
        TropicalCyclone(
            storm_name="Plate_Fleet_1715",
            year=1715,
            eye_lat=27.5,
            eye_lon=-80.0,
            v_max_m_s=50.0,
            p_central_hpa=948.0,
            r_max_km=50.0,
        ),
        # Calm Caribbean baseline
        None,
    ]

    for epoch in range(epochs):
        epoch_loss = 0.0
        for cyc in cyclones:
            optimizer.zero_grad()
            x, edge_attr = evaluator.build_dynamic_features(cyclone=cyc)
            p_detach, scatter_vecs = model(x, evaluator.base_data.edge_index, edge_attr)

            # Target detachment probability: grounded in physical edge features
            # High storm intensity + high hazard level + opposing headwind -> high detachment
            hazard = edge_attr[:, 1:2]
            headwind = torch.clamp(edge_attr[:, 3:4], min=0.0)
            storm_int = edge_attr[:, 4:5]

            target_p = torch.clamp(hazard * 0.4 + storm_int * 0.5 + headwind * 0.3, 0.0, 0.95)

            # Target scatter velocity aligns with storm wind displacement
            target_scatter = torch.cat([storm_int * 1.5, storm_int * -1.0], dim=-1)

            loss = loss_fn(p_detach, target_p) + 0.5 * loss_fn(scatter_vecs, target_scatter)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        loss_history.append(epoch_loss / len(cyclones))

    model.eval()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return loss_history
