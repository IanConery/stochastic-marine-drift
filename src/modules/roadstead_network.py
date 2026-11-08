"""Expansion and management engine for Caribbean colonial roadsteads, submerged ports, and convoy diversion corridors."""

import argparse
from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.utils.geo import haversine_bearing_deg, haversine_distance_m

logger = logging.getLogger(__name__)


@dataclass
class RoadsteadNode:
    """Historical navigation node representing a port, choke point, roadstead, or submerged harbor."""

    node_id: str
    name: str
    node_type: str  # 'port', 'choke_point', 'roadstead', 'submerged_port'
    latitude: float
    longitude: float
    significance: str
    holding_quality: str = "good"  # 'excellent', 'good', 'fair', 'exposed', 'submerged'
    shelter_bearing_deg: Optional[float] = None
    max_vessel_capacity: int = 30

    def to_feature(self) -> Dict[str, Any]:
        """Convert node entity to standard GeoJSON Point Feature."""
        return {
            "type": "Feature",
            "id": f"node_{self.node_id}",
            "geometry": {
                "type": "Point",
                "coordinates": [round(self.longitude, 6), round(self.latitude, 6)],
            },
            "properties": {
                "node_id": self.node_id,
                "name": self.name,
                "type": self.node_type,
                "latitude": round(self.latitude, 6),
                "longitude": round(self.longitude, 6),
                "significance": self.significance,
                "holding_quality": self.holding_quality,
                "shelter_bearing_deg": self.shelter_bearing_deg,
                "max_vessel_capacity": self.max_vessel_capacity,
            },
        }


@dataclass
class CorridorEdge:
    """Maritime navigation corridor connecting two nodes with hydrodynamic and hazard parameters."""

    edge_id: str
    source: str
    target: str
    name: str
    route_type: str  # 'trunk', 'diversion', 'feeder', 'emergency_refuge'
    distance_nmi: float
    hazard_level: float  # 0.0 (benign) to 1.0 (extreme navigational peril)
    prevailing_current_knots: float
    prevailing_wind_heading_deg: float
    coordinates: List[List[float]] = field(default_factory=list)

    def to_feature(self) -> Dict[str, Any]:
        """Convert corridor edge entity to standard GeoJSON LineString Feature."""
        return {
            "type": "Feature",
            "id": f"edge_{self.edge_id}",
            "geometry": {
                "type": "LineString",
                "coordinates": self.coordinates,
            },
            "properties": {
                "edge_id": self.edge_id,
                "source": self.source,
                "target": self.target,
                "name": self.name,
                "route_type": self.route_type,
                "distance_nmi": round(self.distance_nmi, 2),
                "hazard_level": round(self.hazard_level, 2),
                "prevailing_current_knots": round(self.prevailing_current_knots, 2),
                "prevailing_wind_heading_deg": round(self.prevailing_wind_heading_deg, 1),
            },
        }


class RoadsteadNetworkBuilder:
    """Builds, expands, and maintains the Caribbean convoy routing graph with historical roadsteads."""

    # Historical expansion nodes: submerged ports and hurricane roadsteads
    EXPANSION_NODES: List[RoadsteadNode] = [
        RoadsteadNode(
            node_id="port_royal",
            name="Port Royal (Submerged Harbor)",
            node_type="submerged_port",
            latitude=17.937,
            longitude=-76.841,
            significance="Major 17th-century British deep-water naval anchorage; two-thirds submerged June 7, 1692 in earthquake/liquefaction",
            holding_quality="submerged",
            shelter_bearing_deg=180.0,
            max_vessel_capacity=100,
        ),
        RoadsteadNode(
            node_id="nombre_de_dios",
            name="Nombre de Dios (Colonial Roadstead)",
            node_type="submerged_port",
            latitude=9.580,
            longitude=-79.470,
            significance="Early 16th-century Atlantic terminus of Camino Real silver route; abandoned 1596 following Drake's assault and heavy riverine siltation",
            holding_quality="submerged",
            shelter_bearing_deg=220.0,
            max_vessel_capacity=20,
        ),
        RoadsteadNode(
            node_id="dry_tortugas",
            name="Dry Tortugas / Garden Key",
            node_type="roadstead",
            latitude=24.628,
            longitude=-82.873,
            significance="Western Florida Straits holding anchorage; strategic lee refuge for the Carrera de Indias during sudden northern winter gales (nortes)",
            holding_quality="excellent",
            shelter_bearing_deg=0.0,
            max_vessel_capacity=40,
        ),
        RoadsteadNode(
            node_id="isle_of_pines",
            name="Isle of Pines (Isla de la Juventud)",
            node_type="roadstead",
            latitude=21.600,
            longitude=-82.850,
            significance="Major southern Cuban hurricane haven and careening bay (Ensenada de la Siguanea); holding refuge before rounding Cape San Antonio into Havana",
            holding_quality="excellent",
            shelter_bearing_deg=90.0,
            max_vessel_capacity=50,
        ),
        RoadsteadNode(
            node_id="roatan",
            name="Roatán (Bay Islands / Port Royal Bay)",
            node_type="roadstead",
            latitude=16.328,
            longitude=-86.533,
            significance="Western Caribbean lee anchorage and careening harbor; deep natural bay protected from trade wind swells and Gulf of Honduras reefs",
            holding_quality="good",
            shelter_bearing_deg=45.0,
            max_vessel_capacity=25,
        ),
        RoadsteadNode(
            node_id="virgin_gorda",
            name="Virgin Gorda (Sir Francis Drake Channel)",
            node_type="roadstead",
            latitude=18.490,
            longitude=-64.430,
            significance="Deep-water sheltered sound guarding Anegada Passage entrance; classic colonial staging harbor and hurricane hole before transatlantic transit",
            holding_quality="excellent",
            shelter_bearing_deg=135.0,
            max_vessel_capacity=60,
        ),
        RoadsteadNode(
            node_id="saint_eustatius",
            name="Sint Eustatius (Oranjestad Roads)",
            node_type="roadstead",
            latitude=17.483,
            longitude=-62.983,
            significance="Premier 17th-18th century neutral transshipment roadstead ('The Golden Rock') in the Leeward Islands; open leeward roadstead",
            holding_quality="fair",
            shelter_bearing_deg=270.0,
            max_vessel_capacity=80,
        ),
    ]

    # Historical expansion edges: intermediate diversion corridors and emergency refuge legs
    EXPANSION_EDGES: List[Dict[str, Any]] = [
        # Feeder / coastal link: Portobelo <-> Nombre de Dios
        {
            "edge_id": "portobelo_nombre_de_dios",
            "source": "portobelo",
            "target": "nombre_de_dios",
            "name": "Panama North Coast Feeder (Portobelo to Nombre de Dios)",
            "route_type": "feeder",
            "hazard_level": 0.40,
            "prevailing_current_knots": 0.8,
            "prevailing_wind_heading_deg": 260.0,
        },
        # Trans-Caribbean route: Cartagena -> Port Royal (Jamaica)
        {
            "edge_id": "cartagena_port_royal",
            "source": "cartagena",
            "target": "port_royal",
            "name": "Central Caribbean Passage (Cartagena to Port Royal)",
            "route_type": "diversion",
            "hazard_level": 0.45,
            "prevailing_current_knots": 1.2,
            "prevailing_wind_heading_deg": 280.0,
        },
        # Windward Passage -> Port Royal approach
        {
            "edge_id": "windward_passage_port_royal",
            "source": "windward_passage",
            "target": "port_royal",
            "name": "Jamaica Channel Southbound (Windward Passage to Port Royal)",
            "route_type": "diversion",
            "hazard_level": 0.35,
            "prevailing_current_knots": 1.0,
            "prevailing_wind_heading_deg": 270.0,
        },
        # Port Royal -> Yucatan Channel leeward westbound corridor
        {
            "edge_id": "port_royal_yucatan_channel",
            "source": "port_royal",
            "target": "yucatan_channel",
            "name": "Western Caribbean Leeward Corridor (Port Royal to Yucatan Channel)",
            "route_type": "diversion",
            "hazard_level": 0.50,
            "prevailing_current_knots": 1.5,
            "prevailing_wind_heading_deg": 290.0,
        },
        # Emergency storm diversion: Yucatan Channel -> Isle of Pines refuge
        {
            "edge_id": "yucatan_isle_of_pines",
            "source": "yucatan_channel",
            "target": "isle_of_pines",
            "name": "South Cuba Hurricane Refuge Diversion (Yucatan to Isle of Pines)",
            "route_type": "emergency_refuge",
            "hazard_level": 0.65,
            "prevailing_current_knots": 1.1,
            "prevailing_wind_heading_deg": 310.0,
        },
        # Isle of Pines refuge -> Havana (rounding Cape San Antonio)
        {
            "edge_id": "isle_of_pines_havana",
            "source": "isle_of_pines",
            "target": "havana",
            "name": "Cape San Antonio Transit (Isle of Pines to Havana)",
            "route_type": "diversion",
            "hazard_level": 0.55,
            "prevailing_current_knots": 1.8,
            "prevailing_wind_heading_deg": 340.0,
        },
        # Yucatan Channel -> Roatan Bay Islands lee diversion
        {
            "edge_id": "yucatan_roatan",
            "source": "yucatan_channel",
            "target": "roatan",
            "name": "Gulf of Honduras Lee Diversion (Yucatan Channel to Roatan)",
            "route_type": "emergency_refuge",
            "hazard_level": 0.60,
            "prevailing_current_knots": 1.3,
            "prevailing_wind_heading_deg": 250.0,
        },
        # Florida Straits Marquesas <-> Dry Tortugas holding detour
        {
            "edge_id": "marquesas_dry_tortugas",
            "source": "florida_straits_marquesas",
            "target": "dry_tortugas",
            "name": "Tortugas Bank Holding Diversion (Marquesas to Dry Tortugas)",
            "route_type": "emergency_refuge",
            "hazard_level": 0.70,
            "prevailing_current_knots": 2.2,
            "prevailing_wind_heading_deg": 045.0,
        },
        # Dry Tortugas -> Havana cross-Straits corridor
        {
            "edge_id": "dry_tortugas_havana",
            "source": "dry_tortugas",
            "target": "havana",
            "name": "Florida Straits Southbound Transit (Dry Tortugas to Havana)",
            "route_type": "diversion",
            "hazard_level": 0.65,
            "prevailing_current_knots": 2.5,
            "prevailing_wind_heading_deg": 030.0,
        },
        # Mona Passage -> Virgin Gorda (Sir Francis Drake Channel)
        {
            "edge_id": "mona_virgin_gorda",
            "source": "mona_passage",
            "target": "virgin_gorda",
            "name": "Virgin Passage Inshore Corridor (Mona Passage to Virgin Gorda)",
            "route_type": "diversion",
            "hazard_level": 0.35,
            "prevailing_current_knots": 0.9,
            "prevailing_wind_heading_deg": 080.0,
        },
        # Virgin Gorda -> Sint Eustatius roadstead link
        {
            "edge_id": "virgin_gorda_saint_eustatius",
            "source": "virgin_gorda",
            "target": "saint_eustatius",
            "name": "Anegada-Leeward Roadstead Link (Virgin Gorda to St. Eustatius)",
            "route_type": "feeder",
            "hazard_level": 0.30,
            "prevailing_current_knots": 0.8,
            "prevailing_wind_heading_deg": 095.0,
        },
        # Virgin Gorda -> Bermuda Approach Atlantic homeward leg
        {
            "edge_id": "virgin_gorda_bermuda",
            "source": "virgin_gorda",
            "target": "bermuda_approach",
            "name": "Anegada Passage Transatlantic Departure (Virgin Gorda to Bermuda)",
            "route_type": "trunk",
            "hazard_level": 0.40,
            "prevailing_current_knots": 0.7,
            "prevailing_wind_heading_deg": 060.0,
        },
    ]

    @classmethod
    def load_base_network(cls, geojson_path: Path) -> Tuple[Dict[str, RoadsteadNode], List[CorridorEdge]]:
        """Load base network GeoJSON and parse into typed Node and Edge structures."""
        if not geojson_path.exists():
            raise FileNotFoundError(f"Base route network not found at: {geojson_path}")

        with open(geojson_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        nodes: Dict[str, RoadsteadNode] = {}
        edges: List[CorridorEdge] = []

        for feat in data.get("features", []):
            geom_type = feat.get("geometry", {}).get("type")
            props = feat.get("properties", {})

            if geom_type == "Point":
                lon, lat = feat["geometry"]["coordinates"]
                nid = props["node_id"]
                nodes[nid] = RoadsteadNode(
                    node_id=nid,
                    name=props.get("name", nid),
                    node_type=props.get("type", "waypoint"),
                    latitude=lat,
                    longitude=lon,
                    significance=props.get("significance", ""),
                    holding_quality=props.get("holding_quality", "good"),
                    shelter_bearing_deg=props.get("shelter_bearing_deg"),
                    max_vessel_capacity=props.get("max_vessel_capacity", 30),
                )
            elif geom_type == "LineString":
                src = props["source"]
                tgt = props["target"]
                eid = props.get("edge_id", f"{src}_{tgt}")
                edges.append(
                    CorridorEdge(
                        edge_id=eid,
                        source=src,
                        target=tgt,
                        name=props.get("name", f"{src} to {tgt}"),
                        route_type=props.get("route_type", "trunk"),
                        distance_nmi=float(props.get("distance_nmi", 100.0)),
                        hazard_level=float(props.get("hazard_level", 0.3)),
                        prevailing_current_knots=float(props.get("prevailing_current_knots", 1.0)),
                        prevailing_wind_heading_deg=float(props.get("prevailing_wind_heading_deg", 270.0)),
                        coordinates=feat["geometry"]["coordinates"],
                    )
                )

        return nodes, edges

    @classmethod
    def build_expanded_network(cls, base_geojson_path: Path) -> Dict[str, Any]:
        """Merge baseline network with historical roadsteads, submerged ports, and diversion edges."""
        nodes, edges = cls.load_base_network(base_geojson_path)

        # 1. Add expansion nodes
        for exp_node in cls.EXPANSION_NODES:
            if exp_node.node_id not in nodes:
                nodes[exp_node.node_id] = exp_node
                logger.info(f"Added roadstead node: {exp_node.node_id} ({exp_node.name})")

        # 2. Add expansion edges with geodesic distance and coordinates
        existing_edge_ids = {e.edge_id for e in edges}

        for edge_spec in cls.EXPANSION_EDGES:
            eid = edge_spec["edge_id"]
            src_id = edge_spec["source"]
            tgt_id = edge_spec["target"]

            if eid in existing_edge_ids:
                continue

            if src_id not in nodes or tgt_id not in nodes:
                logger.warning(f"Skipping edge {eid}: endpoints {src_id} -> {tgt_id} not both in nodes.")
                continue

            src_node = nodes[src_id]
            tgt_node = nodes[tgt_id]

            # Compute geodesic distance in nautical miles (1 nmi = 1852 meters)
            dist_m = haversine_distance_m(
                src_node.latitude, src_node.longitude, tgt_node.latitude, tgt_node.longitude
            )
            dist_nmi = dist_m / 1852.0

            # LineString coordinates [[lon1, lat1], [lon2, lat2]]
            coords = [
                [round(src_node.longitude, 6), round(src_node.latitude, 6)],
                [round(tgt_node.longitude, 6), round(tgt_node.latitude, 6)],
            ]

            corridor = CorridorEdge(
                edge_id=eid,
                source=src_id,
                target=tgt_id,
                name=edge_spec["name"],
                route_type=edge_spec["route_type"],
                distance_nmi=dist_nmi,
                hazard_level=edge_spec["hazard_level"],
                prevailing_current_knots=edge_spec["prevailing_current_knots"],
                prevailing_wind_heading_deg=edge_spec["prevailing_wind_heading_deg"],
                coordinates=coords,
            )
            edges.append(corridor)
            existing_edge_ids.add(eid)
            logger.info(f"Added corridor edge: {eid} ({src_id} -> {tgt_id}, {dist_nmi:.1f} nmi)")

        # 3. Assemble GeoJSON FeatureCollection
        features = []
        for n in nodes.values():
            features.append(n.to_feature())
        for e in edges:
            features.append(e.to_feature())

        return {
            "type": "FeatureCollection",
            "name": "caribbean_convoy_routes_expanded",
            "description": "Historical Carrera de Indias maritime trade routes, hurricane roadsteads, submerged ports, and diversion corridors (16th-18th Century).",
            "features": features,
        }

    @classmethod
    def export_geojson(cls, network_geojson: Dict[str, Any], output_path: Path) -> None:
        """Export GeoJSON FeatureCollection to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(network_geojson, f, indent=2)
        logger.info(f"Network exported to {output_path}")

    @classmethod
    def generate_summary_markdown(cls, network_geojson: Dict[str, Any], output_path: Path) -> None:
        """Generate academic Markdown summary detailing all network nodes, roadsteads, and corridors."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        for feat in network_geojson["features"]:
            if feat["geometry"]["type"] == "Point":
                nodes.append(feat["properties"])
            elif feat["geometry"]["type"] == "LineString":
                edges.append(feat["properties"])

        lines = [
            "# Historical Caribbean Convoy & Roadstead Navigation Network Summary",
            "",
            f"Total Navigational Nodes: **{len(nodes)}**  ",
            f"Total Maritime Corridors & Diversion Edges: **{len(edges)}**  ",
            "",
            "## 1. Navigational Nodes & Anchorages",
            "| Node ID | Name | Classification | Coordinates (WGS84) | Holding Quality | Max Capacity | Strategic Role |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
        ]

        for n in sorted(nodes, key=lambda x: (x.get("type", ""), x.get("name", ""))):
            coords = f"`{n['latitude']:.3f}°N, {abs(n['longitude']):.3f}°W`"
            lines.append(
                f"| `{n['node_id']}` | **{n['name']}** | `{n['type']}` | {coords} | {n.get('holding_quality', 'good')} | {n.get('max_vessel_capacity', 30)} | {n.get('significance', '')[:80]}... |"
            )

        lines.extend([
            "",
            "## 2. Convoy Corridors & Diversion Edges",
            "| Edge ID | Source -> Target | Corridor Name | Type | Distance (nmi) | Hazard Level | Current (kn) | Wind Heading |",
            "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |",
        ])

        for e in sorted(edges, key=lambda x: (x.get("route_type", ""), x.get("name", ""))):
            endpoints = f"`{e['source']}` -> `{e['target']}`"
            lines.append(
                f"| `{e['edge_id']}` | {endpoints} | {e['name']} | `{e.get('route_type', 'trunk')}` | {e['distance_nmi']:.1f} | {e['hazard_level']:.2f} | {e['prevailing_current_knots']:.1f} | {e['prevailing_wind_heading_deg']:.0f}° |"
            )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        logger.info(f"Summary markdown written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expand Caribbean Convoy Network with Roadsteads and Submerged Ports")
    parser.add_argument(
        "--base_geojson",
        default="data/networks/caribbean_convoy_routes.geojson",
        help="Path to baseline convoy routes GeoJSON",
    )
    parser.add_argument(
        "--output_geojson",
        default="data/networks/caribbean_convoy_routes.geojson",
        help="Path to write expanded convoy routes GeoJSON",
    )
    parser.add_argument(
        "--summary_md",
        default="data/processed/convoy_roadstead_network_summary.md",
        help="Path to write Markdown summary report",
    )
    args = parser.parse_args()

    builder = RoadsteadNetworkBuilder()
    expanded_geojson = builder.build_expanded_network(Path(args.base_geojson))
    builder.export_geojson(expanded_geojson, Path(args.output_geojson))
    builder.generate_summary_markdown(expanded_geojson, Path(args.summary_md))

    print(
        f"Network expansion complete. Total features: {len(expanded_geojson['features'])} "
        f"({sum(1 for f in expanded_geojson['features'] if f['geometry']['type'] == 'Point')} nodes, "
        f"{sum(1 for f in expanded_geojson['features'] if f['geometry']['type'] == 'LineString')} edges)."
    )
