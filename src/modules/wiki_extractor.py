"""High-throughput streaming extractor for historical maritime wreck records and colonial roadsteads from Wikipedia archives."""

import argparse
import bz2
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, Generator, List, Optional, Set, Tuple
import urllib.parse
import xml.etree.ElementTree as ET

import numpy as np

from src.profiles.vessel_class import MacroArchetype, RigType

logger = logging.getLogger(__name__)


class TaphonomicExposure(str, Enum):
    """Hydrodynamic exposure classification for seafloor taphonomic calibration."""

    SHELTERED_DEEP = "sheltered_deep"  # Depth > 60m; bottom wave orbital velocity ub ≈ 0; pure backtracker baseline
    SHELTERED_TOPOGRAPHIC = "sheltered_topographic"  # Enclosed sounds, bays, roadsteads with restricted wave fetch
    EXPOSED_HIGH_ENERGY = "exposed_high_energy"  # Shallow barrier reefs, shoals (d <= 25m); high wave mobilization


class CasualtyFate(str, Enum):
    """Categorical casualty outcome for modeling survivorship bias and hazard rates."""

    FOUNDERED_DEEP = "foundered_deep"  # Catastrophic structural loss in open/deep water (event = 1)
    GROUNDED_REEF = "grounded_reef"  # Driven onto barrier reef, rock, or shoal (event = 1)
    BEACHED_INTENTIONAL = "beached_intentional"  # Deliberately run aground to save crew (right-censored survival)
    SURVIVED_DAMAGED = "survived_damaged"  # Jettisoned cargo, dismasted, made port (right-censored survival)
    SUBMERGED_PORT = "submerged_port"  # Catastrophic seismic/geological port inundation (e.g. Port Royal)


@dataclass
class ExtractedWreckCandidate:
    """Standardized historical maritime casualty record extracted from archival text."""

    title: str
    year: Optional[int]
    macro_archetype: MacroArchetype
    rig_type: RigType
    mast_count: int
    lat: Optional[float]
    lon: Optional[float]
    water_depth_m: Optional[float]
    casualty_fate: CasualtyFate
    is_censored_survival: bool
    taphonomic_exposure: TaphonomicExposure
    tonnage: Optional[float]
    armament_guns: Optional[int]
    nationality: Optional[str]
    extract_snippet: str
    categories: List[str] = field(default_factory=list)
    source_title: str = ""
    full_text: str = ""

    # Archival Provenance & Version Lineage
    wiki_url: str = ""
    dump_source: str = "enwiki-2026-06-01"
    dump_filename: str = "wiki_dump.json.bz2"
    dump_year: int = 2026
    archive_line: Optional[int] = None
    local_path: Optional[str] = None
    partition: str = "train"

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a clean dictionary."""
        d = asdict(self)
        d["macro_archetype"] = self.macro_archetype.value
        d["rig_type"] = self.rig_type.value
        d["casualty_fate"] = self.casualty_fate.value
        d["taphonomic_exposure"] = self.taphonomic_exposure.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExtractedWreckCandidate":
        """Reconstruct candidate from clean dictionary."""
        return cls(
            title=d["title"],
            year=d.get("year"),
            macro_archetype=MacroArchetype(d["macro_archetype"]),
            rig_type=RigType(d["rig_type"]),
            mast_count=d.get("mast_count", 3),
            lat=d.get("lat"),
            lon=d.get("lon"),
            water_depth_m=d.get("water_depth_m"),
            casualty_fate=CasualtyFate(d["casualty_fate"]),
            is_censored_survival=d.get("is_censored_survival", False),
            taphonomic_exposure=TaphonomicExposure(d["taphonomic_exposure"]),
            tonnage=d.get("tonnage"),
            armament_guns=d.get("armament_guns"),
            nationality=d.get("nationality"),
            extract_snippet=d.get("extract_snippet", ""),
            categories=d.get("categories", []),
            source_title=d.get("source_title", ""),
            full_text=d.get("full_text", ""),
            wiki_url=d.get("wiki_url", ""),
            dump_source=d.get("dump_source", "enwiki-2026-06-01"),
            dump_filename=d.get("dump_filename", "wiki_dump.json.bz2"),
            dump_year=d.get("dump_year", 2026),
            archive_line=d.get("archive_line"),
            local_path=d.get("local_path"),
            partition=d.get("partition", "train"),
        )

    def to_geojson_feature(self) -> Optional[Dict[str, Any]]:
        """Convert record to a GeoJSON Feature if valid coordinates exist."""
        if self.lat is None or self.lon is None:
            return None
        return {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [round(self.lon, 6), round(self.lat, 6)],
            },
            "properties": {
                "title": self.title,
                "article_name": self.title,
                "wiki_url": self.wiki_url,
                "local_path": self.local_path,
                "dump_source": self.dump_source,
                "dump_filename": self.dump_filename,
                "dump_year": self.dump_year,
                "archive_line": self.archive_line,
                "partition": self.partition,
                "year": self.year,
                "macro_archetype": self.macro_archetype.value,
                "rig_type": self.rig_type.value,
                "mast_count": self.mast_count,
                "water_depth_m": self.water_depth_m,
                "casualty_fate": self.casualty_fate.value,
                "is_censored_survival": self.is_censored_survival,
                "taphonomic_exposure": self.taphonomic_exposure.value,
                "tonnage": self.tonnage,
                "armament_guns": self.armament_guns,
                "nationality": self.nationality,
            },
        }


class WikiMaritimeExtractor:
    """Extracts, filters, and structures maritime casualties from compressed Wikipedia archives."""

    # Caribbean, Gulf of Mexico, Florida Straits & Western Atlantic Bounding Box
    DEFAULT_MIN_LAT = 9.0
    DEFAULT_MAX_LAT = 35.0
    DEFAULT_MIN_LON = -98.0
    DEFAULT_MAX_LON = -55.0

    # Regional keyword anchors
    REGIONAL_KEYWORDS = {
        "caribbean", "florida", "bahamas", "cuba", "havana", "jamaica",
        "hispaniola", "puerto rico", "gulf of mexico", "bermuda", "antilles",
        "barbados", "haiti", "virgin islands", "yucatan", "belize", "honduras",
        "cartagena", "veracruz", "panama", "portobelo", "key west", "marquesas",
        "dry tortugas", "crooked island", "silver bank", "quicksands", "looe key"
    }

    # Vessel taxonomy regex rules
    ARCHETYPE_PATTERNS = [
        (re.compile(r"\b(galleon|ship of the line|first[- ]rate|second[- ]rate|third[- ]rate|fourth[- ]rate|fifth[- ]rate|frigate|nao|fluyt|urca|carrack|man[- ]of[- ]war)\b", re.IGNORECASE), MacroArchetype.THREE_MAST_SQUARE, RigType.SQUARE, 3),
        (re.compile(r"\b(brig|snow|brigantine)\b", re.IGNORECASE), MacroArchetype.TWO_MAST_SQUARE, RigType.SQUARE, 2),
        (re.compile(r"\b(schooner|ketch|bilander)\b", re.IGNORECASE), MacroArchetype.TWO_MAST_FORE_AFT, RigType.FORE_AND_AFT, 2),
        (re.compile(r"\b(sloop|cutter|shallop|pennace|pinnace)\b", re.IGNORECASE), MacroArchetype.ONE_MAST_FORE_AFT, RigType.FORE_AND_AFT, 1),
        (re.compile(r"\b(steamship|steamer|container ship|freighter|tanker|bulk carrier|ro-ro)\b", re.IGNORECASE), MacroArchetype.MODERN_UNMASTED, RigType.UNMASTED_POWER, 0),
    ]

    # Coordinate regex patterns
    COORD_PATTERN_DMS = re.compile(
        r"(\d{1,2})[°\s]+(\d{1,2})[′'\s]*([0-9.]+)?[\"″]?\s*([NS])[\s,]+(\d{1,3})[°\s]+(\d{1,2})[′'\s]*([0-9.]+)?[\"″]?\s*([EW])",
        re.IGNORECASE,
    )
    COORD_PATTERN_DECIMAL = re.compile(
        r"([-+]?\d{1,2}\.\d{2,6})[\s,]+([-+]?\d{1,3}\.\d{2,6})",
        re.IGNORECASE,
    )
    COORD_PATTERN_WIKI_TEMPLATE = re.compile(
        r"\{\{coord\|(\d{1,2})\|(\d{1,2})\|([0-9.]+)?\|?([NS])\|(\d{1,3})\|(\d{1,2})\|([0-9.]+)?\|?([EW])",
        re.IGNORECASE,
    )

    def __init__(
        self,
        min_lat: float = DEFAULT_MIN_LAT,
        max_lat: float = DEFAULT_MAX_LAT,
        min_lon: float = DEFAULT_MIN_LON,
        max_lon: float = DEFAULT_MAX_LON,
    ) -> None:
        """Initialize maritime extractor with spatial bounding constraints."""
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon

    def extract_coordinates(self, text: str) -> Tuple[Optional[float], Optional[float]]:
        """Extract WGS84 latitude and longitude in decimal degrees from article text or infobox."""
        # 1. Try {{coord|lat_d|lat_m|...}} Wikipedia template
        match_tmpl = self.COORD_PATTERN_WIKI_TEMPLATE.search(text)
        if match_tmpl:
            lat_d, lat_m, lat_s, lat_dir, lon_d, lon_m, lon_s, lon_dir = match_tmpl.groups()
            lat = float(lat_d) + float(lat_m) / 60.0 + (float(lat_s or 0.0) / 3600.0)
            if lat_dir.upper() == "S":
                lat = -lat
            lon = float(lon_d) + float(lon_m) / 60.0 + (float(lon_s or 0.0) / 3600.0)
            if lon_dir.upper() == "W":
                lon = -lon
            return lat, lon

        # 2. Try Degrees Minutes Seconds (DMS)
        match_dms = self.COORD_PATTERN_DMS.search(text)
        if match_dms:
            lat_d, lat_m, lat_s, lat_dir, lon_d, lon_m, lon_s, lon_dir = match_dms.groups()
            lat = float(lat_d) + float(lat_m) / 60.0 + (float(lat_s or 0.0) / 3600.0)
            if lat_dir.upper() == "S":
                lat = -lat
            lon = float(lon_d) + float(lon_m) / 60.0 + (float(lon_s or 0.0) / 3600.0)
            if lon_dir.upper() == "W":
                lon = -lon
            return lat, lon

        # 3. Try standard Decimal notation
        for match_dec in self.COORD_PATTERN_DECIMAL.finditer(text):
            lat_val = float(match_dec.group(1))
            lon_val = float(match_dec.group(2))
            if -90.0 <= lat_val <= 90.0 and -180.0 <= lon_val <= 180.0:
                if self.min_lat <= lat_val <= self.max_lat and self.min_lon <= lon_val <= self.max_lon:
                    return lat_val, lon_val

        return None, None

    def extract_year(self, title: str, text: str, categories: List[str]) -> Optional[int]:
        """Extract verified loss or incident year."""
        # Check categories first (e.g., '1622 shipwrecks', 'Maritime incidents in 1744')
        cat_year = re.search(r"\b(1[5-9]\d{2}|20[0-2]\d)\b", " ".join(categories))
        if cat_year:
            return int(cat_year.group(1))

        # Check title (e.g., 'San Jose (1708)', '1715 Treasure Fleet')
        title_year = re.search(r"\b(1[5-9]\d{2}|20[0-2]\d)\b", title)
        if title_year:
            return int(title_year.group(1))

        # Check first 500 characters of extract text
        text_year = re.search(r"\b(?:in|during|of|on)\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)?\s*(1[5-9]\d{2}|20[0-2]\d)\b", text[:500], re.IGNORECASE)
        if text_year:
            return int(text_year.group(1))

        return None

    def classify_vessel_archetype(self, title: str, text: str) -> Tuple[MacroArchetype, RigType, int]:
        """Classify vessel into macro-archetype, rig type, and mast count."""
        corpus_snippet = (title + " " + text[:1000]).lower()
        for pattern, archetype, rig, masts in self.ARCHETYPE_PATTERNS:
            if pattern.search(corpus_snippet):
                return archetype, rig, masts

        # Default fallback: 3-mast square historical sailing hull
        return MacroArchetype.THREE_MAST_SQUARE, RigType.SQUARE, 3

    def classify_fate_and_survivorship(self, text: str, categories: List[str]) -> Tuple[CasualtyFate, bool]:
        """Classify casualty fate and determine if observation is right-censored (survived/beached)."""
        corpus = (text[:1500] + " " + " ".join(categories)).lower()

        # Seismic submerged port (e.g. Port Royal)
        if "submerged" in corpus and ("port royal" in corpus or "earthquake" in corpus):
            return CasualtyFate.SUBMERGED_PORT, False

        # Right-censored: Deliberate beaching / run ashore to save crew
        if any(w in corpus for w in ["beached", "run ashore", "driven ashore intentionally", "stranded intentionally"]):
            return CasualtyFate.BEACHED_INTENTIONAL, True

        # Right-censored: Survived damaged (jettisoned cargo, dismasted, made port)
        if any(w in corpus for w in ["jettisoned", "made port", "towed to port", "refloated", "repaired", "survived the storm", "limped into"]):
            return CasualtyFate.SURVIVED_DAMAGED, True

        # Reef collision / grounding
        if any(w in corpus for w in ["struck a reef", "grounded on", "wrecked on the reef", "shoal", "breakers"]):
            return CasualtyFate.GROUNDED_REEF, False

        # Foundered in open / deep water
        return CasualtyFate.FOUNDERED_DEEP, False

    def classify_taphonomic_exposure(
        self,
        water_depth_m: Optional[float],
        text: str,
        lat: Optional[float],
        lon: Optional[float],
    ) -> TaphonomicExposure:
        """Classify site hydrodynamic exposure for taphonomic and sediment calibration."""
        # Deep water sites (> 60m) have negligible storm wave bottom orbital velocity
        if water_depth_m is not None and water_depth_m > 60.0:
            return TaphonomicExposure.SHELTERED_DEEP

        corpus = text[:1000].lower()

        # Enclosed sounds, bays, roadsteads, and harbors offer topographic wave dissipation
        if any(w in corpus for w in ["harbor", "harbour", "bay", "sound", "lagoon", "roadstead", "road", "anchorage"]):
            return TaphonomicExposure.SHELTERED_TOPOGRAPHIC

        # Shallow coastal sites (<= 25m) without topographic shelter experience direct wave energy
        if water_depth_m is not None and water_depth_m <= 25.0:
            return TaphonomicExposure.EXPOSED_HIGH_ENERGY

        # Deep ocean terms
        if any(w in corpus for w in ["abyss", "deep water", "abyssal", "ocean floor", "deep ocean", "trench"]):
            return TaphonomicExposure.SHELTERED_DEEP

        # Default shallow open reef/shoal
        return TaphonomicExposure.EXPOSED_HIGH_ENERGY

    def extract_tonnage_and_depth(self, text: str) -> Tuple[Optional[float], Optional[float]]:
        """Extract displacement tonnage and water depth in meters from text."""
        tonnage: Optional[float] = None
        water_depth: Optional[float] = None

        # Tonnage extraction
        tons_match = re.search(r"\b(\d{2,5})\s*(?:tons|toneladas|tonnes|bm)\b", text[:1000], re.IGNORECASE)
        if tons_match:
            try:
                tonnage = float(tons_match.group(1))
            except ValueError:
                pass

        # Depth extraction (feet or meters)
        depth_m_match = re.search(r"\b(\d{1,4}(?:\.\d+)?)\s*(?:m|meters|metres)\s*(?:deep|depth|of water)\b", text[:1500], re.IGNORECASE)
        if depth_m_match:
            try:
                water_depth = float(depth_m_match.group(1))
            except ValueError:
                pass
        else:
            depth_ft_match = re.search(r"\b(\d{1,5})\s*(?:feet|ft)\s*(?:deep|depth|of water)\b", text[:1500], re.IGNORECASE)
            if depth_ft_match:
                try:
                    water_depth = float(depth_ft_match.group(1)) * 0.3048
                except ValueError:
                    pass

        return tonnage, water_depth

    def filter_article(
        self,
        article: Dict[str, Any],
        line_idx: Optional[int] = None,
        dump_name: str = "wiki_dump.json.bz2",
        dump_source: str = "enwiki-2026-06-01",
        dump_year: int = 2026,
    ) -> Optional[ExtractedWreckCandidate]:
        """Evaluate raw JSON Wikipedia article record and return structured candidate if relevant."""
        title = article.get("title", "")
        extract = article.get("extract", "")
        categories = article.get("categories", [])
        cats_str = " ".join(categories).lower()
        full_text = (title + " " + cats_str + " " + extract[:2000]).lower()

        # 1. Reject disambiguation pages, days, lists, and broad overview topics
        title_lower = title.lower()
        if any(title_lower.startswith(p) for p in [
            "economy of", "geography of", "history of", "politics of", "culture of",
            "fauna of", "flora of", "climate of", "list of", "outline of"
        ]):
            return None
        if title_lower in ["atlantic ocean", "caribbean sea", "gulf of mexico"]:
            return None
        if (
            any("disambiguation" in c for c in cats_str.split())
            or "may refer to:" in extract[:600].lower()
            or "{{disambiguation" in extract[:600].lower()
            or "{{wiktionary" in extract[:600].lower()
        ):
            return None
        if any(c.lower().startswith("days of ") for c in categories):
            return None

        # 2. Reject human biography unless explicit ship infobox is present
        has_ship_infobox = any(
            ib in extract[:5000].lower()
            for ib in ["{{infobox ship", "{{infobox shipwreck", "{{infobox watercraft"]
        )
        has_bio_cats = any("births" in c.lower() for c in categories) and any("deaths" in c.lower() for c in categories)
        if has_bio_cats and not has_ship_infobox:
            return None

        # 3. Must relate to maritime casualty, shipwreck, fleet, or historical naval action
        is_maritime_casualty = (
            has_ship_infobox
            or any(
                w in cats_str
                for w in [
                    "shipwreck", "maritime incident", "ships sinking", "ships sunk", "ships lost",
                    "naval battle", "treasure fleet", "grounded", "foundered", "lost at sea",
                    "age of sail shipwrecks", "ships of the line", "galleon", "frigate", "flota"
                ]
            )
            or any(
                p in title
                for p in [
                    "(ship)", "(shipwreck)", "(galleon)", "(frigate)", "(sloop)", "(schooner)",
                    "HMS ", "USS ", "Spanish ship ", "French ship ", "SS ", "RMS ", "SV ",
                    "Nuestra Señora", "San José"
                ]
            )
            or any(
                w in full_text
                for w in [
                    "shipwreck", "treasure fleet", "grounded on a reef", "struck a reef and sank",
                    "foundered in a hurricane", "lost with all hands"
                ]
            )
        )
        if not is_maritime_casualty:
            return None

        # 4. Must match Caribbean / Western Atlantic regional keywords or have valid coords
        lat, lon = self.extract_coordinates(extract)
        has_regional_keyword = any(k in full_text for k in self.REGIONAL_KEYWORDS)

        if lat is not None and lon is not None:
            if not (self.min_lat <= lat <= self.max_lat and self.min_lon <= lon <= self.max_lon):
                return None  # Coordinates outside target Caribbean basin
        elif not has_regional_keyword:
            return None  # No valid coords and no regional keyword

        # 5. Extract and parse parameters
        year = self.extract_year(title, extract, categories)
        archetype, rig, masts = self.classify_vessel_archetype(title, extract)
        fate, is_censored = self.classify_fate_and_survivorship(extract, categories)
        tonnage, depth = self.extract_tonnage_and_depth(extract)
        exposure = self.classify_taphonomic_exposure(depth, extract, lat, lon)

        # Nationality detection
        nationality = None
        if any(w in full_text for w in ["spanish", "spain", "armada", "carrera de indias"]):
            nationality = "Spanish"
        elif any(w in full_text for w in ["british", "royal navy", "england", "hms "]):
            nationality = "British"
        elif any(w in full_text for w in ["french", "france"]):
            nationality = "French"
        elif any(w in full_text for w in ["dutch", "netherlands", "voc"]):
            nationality = "Dutch"
        elif any(w in full_text for w in ["american", "united states", "uss "]):
            nationality = "American"

        wiki_title_slug = urllib.parse.quote(title.replace(" ", "_"))
        wiki_url = f"https://en.wikipedia.org/wiki/{wiki_title_slug}"
        safe_slug = re.sub(r"[^\w\-]", "_", title.lower()).strip("_")
        local_path = f"data/raw/articles/{safe_slug}.json"

        return ExtractedWreckCandidate(
            title=title,
            year=year,
            macro_archetype=archetype,
            rig_type=rig,
            mast_count=masts,
            lat=lat,
            lon=lon,
            water_depth_m=round(depth, 1) if depth else None,
            casualty_fate=fate,
            is_censored_survival=is_censored,
            taphonomic_exposure=exposure,
            tonnage=tonnage,
            armament_guns=None,
            nationality=nationality,
            extract_snippet=extract[:300].strip(),
            categories=categories[:8],
            source_title=title,
            wiki_url=wiki_url,
            dump_source=dump_source,
            dump_filename=dump_name,
            dump_year=dump_year,
            archive_line=line_idx,
            local_path=local_path,
            full_text=article.get("full_text", extract),
        )

    def stream_extract(
        self,
        dump_path: Path,
        max_candidates: int = 500,
        scan_limit: Optional[int] = None,
        dump_source: str = "enwiki-2026-06-01",
        dump_year: int = 2026,
    ) -> List[ExtractedWreckCandidate]:
        """Stream compressed MediaWiki XML or JSON Lines archive and extract candidate records."""
        candidates: List[ExtractedWreckCandidate] = []
        titles_seen: Set[str] = set()

        logger.info(f"Streaming archive from {dump_path} (scan_limit={scan_limit}, max_candidates={max_candidates})")

        is_xml = dump_path.name.endswith(".xml.bz2") or dump_path.name.endswith(".xml")

        if is_xml:
            articles_scanned = 0
            pages_scanned = 0
            with bz2.open(dump_path, "rb") as f:
                for event, elem in ET.iterparse(f, events=("end",)):
                    if elem.tag.endswith("page"):
                        pages_scanned += 1
                        ns = elem.findtext("{*}ns")
                        redirect = elem.find("{*}redirect")
                        if ns == "0" and redirect is None:
                            articles_scanned += 1
                            title = elem.findtext("{*}title") or ""
                            text = elem.findtext(".//{*}text") or ""
                            cats = re.findall(r"\[\[Category:([^\]|]+)", text)

                            data = {
                                "title": title,
                                "extract": text[:15000],
                                "categories": cats,
                                "full_text": text,
                            }
                            cand = self.filter_article(
                                data,
                                line_idx=pages_scanned,
                                dump_name=dump_path.name,
                                dump_source=dump_source,
                                dump_year=dump_year,
                            )
                            if cand and cand.title not in titles_seen:
                                titles_seen.add(cand.title)
                                candidates.append(cand)
                                logger.info(f"[{dump_path.name}] Extracted: {cand.title} ({cand.year}) - {cand.macro_archetype.value} - {cand.taphonomic_exposure.value}")
                                if len(candidates) >= max_candidates:
                                    elem.clear()
                                    break
                            if articles_scanned % 50000 == 0:
                                logger.info(f"[{dump_path.name}] Scanned {articles_scanned} articles ({len(candidates)} candidates found)")
                            if scan_limit and articles_scanned >= scan_limit:
                                elem.clear()
                                break
                        elem.clear()
        else:
            with bz2.open(dump_path, "rt", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if scan_limit and idx >= scan_limit:
                        break
                    try:
                        data = json.loads(line)
                        cand = self.filter_article(
                            data,
                            line_idx=idx,
                            dump_name=dump_path.name,
                            dump_source=dump_source,
                            dump_year=dump_year,
                        )
                        if cand and cand.title not in titles_seen:
                            titles_seen.add(cand.title)
                            candidates.append(cand)
                            if len(candidates) >= max_candidates:
                                break
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue

        logger.info(f"Extraction complete. Found {len(candidates)} candidate maritime records.")
        return candidates

    @staticmethod
    def partition_train_test(
        candidates: List[ExtractedWreckCandidate],
        train_ratio: float = 0.75,
        seed: int = 42,
    ) -> Tuple[List[ExtractedWreckCandidate], List[ExtractedWreckCandidate]]:
        """Stratified train/test partition preserving macro-archetype and taphonomic exposure proportions."""
        rng = np.random.default_rng(seed)
        shuffled = list(candidates)
        rng.shuffle(shuffled)

        # Stratified buckets based on (archetype, taphonomic_exposure)
        strata: Dict[Tuple[str, str], List[ExtractedWreckCandidate]] = {}
        for c in shuffled:
            key = (c.macro_archetype.value, c.taphonomic_exposure.value)
            strata.setdefault(key, []).append(c)

        train: List[ExtractedWreckCandidate] = []
        test: List[ExtractedWreckCandidate] = []

        for group in strata.values():
            n_train = max(1, int(round(len(group) * train_ratio))) if len(group) > 1 else 1
            for c in group[:n_train]:
                c.partition = "train"
                train.append(c)
            for c in group[n_train:]:
                c.partition = "test"
                test.append(c)

        return train, test

    @staticmethod
    def export_jsonl(candidates: List[ExtractedWreckCandidate], out_path: Path) -> None:
        """Persist candidates to line-delimited JSON."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for c in candidates:
                f.write(json.dumps(c.to_dict()) + "\n")

    @staticmethod
    def export_geojson(candidates: List[ExtractedWreckCandidate], out_path: Path) -> None:
        """Export candidates with valid coordinates as a GeoJSON FeatureCollection."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        features = [c.to_geojson_feature() for c in candidates if c.to_geojson_feature() is not None]
        collection = {"type": "FeatureCollection", "features": features}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(collection, f, indent=2)

    @staticmethod
    def export_article_cache(candidates: List[ExtractedWreckCandidate], out_dir: Path) -> None:
        """Export complete raw wikitext for each candidate into individual JSON files in data/raw/articles/."""
        articles_dir = out_dir / "raw" / "articles" if out_dir.name != "raw" else out_dir / "articles"
        articles_dir.mkdir(parents=True, exist_ok=True)
        for c in candidates:
            safe_slug = re.sub(r"[^\w\-]", "_", c.title.lower()).strip("_")
            article_file = articles_dir / f"{safe_slug}.json"
            record = {
                "title": c.title,
                "wiki_url": c.wiki_url,
                "dump_source": c.dump_source,
                "dump_filename": c.dump_filename,
                "dump_year": c.dump_year,
                "archive_line": c.archive_line,
                "partition": c.partition,
                "year": c.year,
                "macro_archetype": c.macro_archetype.value,
                "rig_type": c.rig_type.value,
                "mast_count": c.mast_count,
                "lat": c.lat,
                "lon": c.lon,
                "water_depth_m": c.water_depth_m,
                "casualty_fate": c.casualty_fate.value,
                "is_censored_survival": c.is_censored_survival,
                "taphonomic_exposure": c.taphonomic_exposure.value,
                "tonnage": c.tonnage,
                "nationality": c.nationality,
                "categories": c.categories,
                "extract_snippet": c.extract_snippet,
                "full_text": c.full_text or c.extract_snippet,
            }
            with open(article_file, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)

    def stream_extract_multiprocess(
        self,
        dump_paths: List[Path],
        max_candidates: int = 500,
        scan_limit: Optional[int] = None,
        max_workers: int = 16,
    ) -> List[ExtractedWreckCandidate]:
        """Stream multiple compressed XML chunks concurrently using ProcessPoolExecutor."""
        tasks = [
            (str(p), max_candidates, scan_limit, "enwiki-2026-06-01", 2026)
            for p in dump_paths
        ]

        logger.info(f"Launching parallel extraction across {len(dump_paths)} dumps with {max_workers} workers.")
        all_candidates: List[ExtractedWreckCandidate] = []
        titles_seen: Set[str] = set()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_dump = {
                executor.submit(_extract_chunk_worker, t): t[0] for t in tasks
            }
            for future in as_completed(future_to_dump):
                dump_file = future_to_dump[future]
                try:
                    cands_dicts = future.result()
                    for d in cands_dicts:
                        c = ExtractedWreckCandidate.from_dict(d)
                        if c.title not in titles_seen:
                            titles_seen.add(c.title)
                            all_candidates.append(c)
                    logger.info(f"Completed {Path(dump_file).name} (total unique candidates: {len(all_candidates)})")
                except Exception as e:
                    logger.error(f"Error processing chunk {dump_file}: {e}")

        logger.info(f"Multiprocess extraction complete. Total candidates harvested: {len(all_candidates)}")
        return all_candidates

    @staticmethod
    def generate_summary_markdown(
        candidates: List[ExtractedWreckCandidate],
        out_path: Path,
    ) -> None:
        """Generate human-readable reference summary table."""
        out_path.parent.mkdir(parents=True, exist_ok=True)

        archetype_counts: Dict[str, int] = {}
        exposure_counts: Dict[str, int] = {}
        fate_counts: Dict[str, int] = {}
        censored_count = sum(1 for c in candidates if c.is_censored_survival)

        for c in candidates:
            archetype_counts[c.macro_archetype.value] = archetype_counts.get(c.macro_archetype.value, 0) + 1
            exposure_counts[c.taphonomic_exposure.value] = exposure_counts.get(c.taphonomic_exposure.value, 0) + 1
            fate_counts[c.casualty_fate.value] = fate_counts.get(c.casualty_fate.value, 0) + 1

        lines = [
            "# Historical Maritime Casualty Catalog Summary",
            "",
            f"Total Documented Records: **{len(candidates)}**  ",
            f"Right-Censored Near-Miss Survivors / Beachings: **{censored_count}**  ",
            f"Wrecks with Confirmed Coordinates: **{sum(1 for c in candidates if c.lat is not None)}**  ",
            "",
            "## 1. Macro-Archetype Distribution",
            "| Macro-Archetype | Count | Percentage |",
            "| :--- | :---: | :---: |",
        ]

        for k, v in sorted(archetype_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (v / len(candidates)) * 100.0 if candidates else 0.0
            lines.append(f"| `{k}` | {v} | {pct:.1f}% |")

        lines.extend([
            "",
            "## 2. Taphonomic Exposure Regimes",
            "| Exposure Classification | Count | Calibration Role |",
            "| :--- | :---: | :--- |",
            f"| `sheltered_deep` | {exposure_counts.get('sheltered_deep', 0)} | Clean-room leeway & 3D glide calibration (zero wave noise) |",
            f"| `sheltered_topographic` | {exposure_counts.get('sheltered_topographic', 0)} | Enclosed harbor/sound wave attenuation |",
            f"| `exposed_high_energy` | {exposure_counts.get('exposed_high_energy', 0)} | Multi-century storm dispersal, Shields saltation & burial |",
            "",
            "## 3. Casualty Fates & Survivorship Outcomes",
            "| Casualty Fate | Count | Model Status |",
            "| :--- | :---: | :--- |",
        ])

        for k, v in sorted(fate_counts.items(), key=lambda x: x[1], reverse=True):
            status = "Right-Censored Survival" if "beached" in k or "survived" in k else "Event Observed"
            lines.append(f"| `{k}` | {v} | {status} |")

        train_count = sum(1 for c in candidates if c.partition == "train")
        test_count = sum(1 for c in candidates if c.partition == "test")
        train_pct = (train_count / len(candidates)) * 100.0 if candidates else 0.0
        test_pct = (test_count / len(candidates)) * 100.0 if candidates else 0.0

        lines.extend([
            "",
            "## 4. Cohort Partitioning (75/25 Stratified Holdout)",
            "| Partition | Count | Percentage | Purpose |",
            "| :--- | :---: | :---: | :--- |",
            f"| `train` | {train_count} | {train_pct:.1f}% | Lagrangian backtracker calibration & survival hazard model fit |",
            f"| `test` | {test_count} | {test_pct:.1f}% | Out-of-sample SAR containment, Harrell's C-index & Brier score evaluation |",
            "",
        ])

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def _extract_chunk_worker(args_tuple: Tuple[str, Optional[int], Optional[int], str, int]) -> List[Dict[str, Any]]:
    """Worker function for concurrent multi-chunk Wikipedia processing."""
    dump_path_str, max_candidates, scan_limit, dump_source, dump_year = args_tuple
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    extractor = WikiMaritimeExtractor()
    cands = extractor.stream_extract(
        Path(dump_path_str),
        max_candidates=max_candidates or 500,
        scan_limit=scan_limit,
        dump_source=dump_source,
        dump_year=dump_year,
    )
    return [c.to_dict() for c in cands]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Extract Caribbean maritime casualties from Wikipedia archive")
    parser.add_argument(
        "--dump_path",
        default="/mnt/storage/wiki",
        help="Path to single enwiki_*.xml.bz2 or directory containing dumps",
    )
    parser.add_argument("--out_dir", default="data", help="Root data directory")
    parser.add_argument("--output_jsonl", default=None, help="Explicit path for output JSONL")
    parser.add_argument("--output_geojson", default=None, help="Explicit path for output GeoJSON")
    parser.add_argument("--summary_md", default=None, help="Explicit path for summary markdown")
    parser.add_argument("--limit", type=int, default=500, help="Maximum candidates to extract")
    parser.add_argument("--scan_limit", type=int, default=None, help="Scan line/page ceiling per chunk for testing")
    parser.add_argument("--workers", type=int, default=16, help="Parallel worker processes for multi-chunk extraction")
    args = parser.parse_args()

    extractor = WikiMaritimeExtractor()
    dump_path = Path(args.dump_path)
    if not dump_path.exists():
        logger.error(f"Dump file/directory not found: {dump_path}")
        sys.exit(1)

    if dump_path.is_dir():
        # Find all enwiki_*.xml.bz2 in natural numerical sort order
        dump_files = sorted(
            dump_path.glob("enwiki_*.xml.bz2"),
            key=lambda p: int(re.search(r"\d+", p.stem).group()) if re.search(r"\d+", p.stem) else 0,
        )
        if not dump_files:
            logger.error(f"No enwiki_*.xml.bz2 files found in directory: {dump_path}")
            sys.exit(1)

        candidates = extractor.stream_extract_multiprocess(
            dump_paths=dump_files,
            max_candidates=args.limit,
            scan_limit=args.scan_limit,
            max_workers=min(args.workers, len(dump_files)),
        )
    else:
        candidates = extractor.stream_extract(
            dump_path=dump_path,
            max_candidates=args.limit,
            scan_limit=args.scan_limit,
        )

    out_dir = Path(args.out_dir)
    jsonl_path = Path(args.output_jsonl) if args.output_jsonl else out_dir / "raw" / "wikipedia_maritime_extracts.jsonl"
    geojson_path = Path(args.output_geojson) if args.output_geojson else out_dir / "processed" / "expanded_shipwreck_catalog.geojson"
    summary_path = Path(args.summary_md) if args.summary_md else out_dir / "processed" / "historical_wreck_summary.md"

    if candidates:
        train_cands, test_cands = extractor.partition_train_test(candidates, train_ratio=0.75, seed=42)
        candidates = train_cands + test_cands

    extractor.export_article_cache(candidates, out_dir)
    extractor.export_jsonl(candidates, jsonl_path)
    extractor.export_geojson(candidates, geojson_path)
    extractor.generate_summary_markdown(candidates, summary_path)

    logger.info(f"Successfully processed {len(candidates)} records.")
    logger.info(f"Full articles cached to: {out_dir / 'raw' / 'articles'}")
    logger.info(f"JSONL exported to: {jsonl_path}")
    logger.info(f"GeoJSON exported to: {geojson_path}")
    logger.info(f"Summary Markdown exported to: {summary_path}")
