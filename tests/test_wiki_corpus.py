"""Unit and integration tests for the Wikipedia maritime casualty extractor."""

import json
from pathlib import Path
import pytest

from src.modules.wiki_extractor import (
    CasualtyFate,
    ExtractedWreckCandidate,
    TaphonomicExposure,
    WikiMaritimeExtractor,
)
from src.profiles.vessel_class import MacroArchetype, RigType


@pytest.fixture
def extractor() -> WikiMaritimeExtractor:
    """Fixture providing initialized extractor with Caribbean bounding coordinates."""
    return WikiMaritimeExtractor(min_lat=9.0, max_lat=35.0, min_lon=-98.0, max_lon=-55.0)


def test_coordinate_parsing_varieties(extractor: WikiMaritimeExtractor):
    """Verify coordinate extraction across Wikipedia template, DMS, and decimal formats."""
    # 1. Wiki coord template
    text_tmpl = "The vessel was lost at {{coord|24|33|15|N|82|17|25|W|type:landmark}} during the gale."
    lat, lon = extractor.extract_coordinates(text_tmpl)
    assert lat is not None and lon is not None
    assert round(lat, 2) == 24.55
    assert round(lon, 2) == -82.29

    # 2. DMS notation
    text_dms = "Grounded on the outer barrier reef at 24° 32′ 48″ N, 81° 24′ 18″ W."
    lat, lon = extractor.extract_coordinates(text_dms)
    assert lat is not None and lon is not None
    assert round(lat, 2) == 24.55
    assert round(lon, 2) == -81.41

    # 3. Decimal notation within bounds
    text_dec = "Wreck position verified by acoustic sonar at 23.5967, -73.8933 in the abyssal plain."
    lat, lon = extractor.extract_coordinates(text_dec)
    assert lat is not None and lon is not None
    assert round(lat, 4) == 23.5967
    assert round(lon, 4) == -73.8933

    # 4. Out-of-bounds coordinates (e.g. English Channel)
    text_oob = "Sunk off Plymouth at 50.37, -4.14."
    lat_oob, lon_oob = extractor.extract_coordinates(text_oob)
    assert lat_oob is None and lon_oob is None


def test_archetype_classification(extractor: WikiMaritimeExtractor):
    """Verify regex-based categorization maps micro-classes to the 5 physical macro-archetypes."""
    # Spanish Galleon
    arch, rig, masts = extractor.classify_vessel_archetype(
        "Nuestra Senora de las Mercedes",
        "A Spanish 34-gun frigate and galleon of the Armada.",
    )
    assert arch == MacroArchetype.THREE_MAST_SQUARE
    assert rig == RigType.SQUARE
    assert masts == 3

    # Two-mast brig
    arch, rig, masts = extractor.classify_vessel_archetype(
        "HMS Badger",
        "An 18-gun brigantine and naval snow of the Royal Navy.",
    )
    assert arch == MacroArchetype.TWO_MAST_SQUARE
    assert rig == RigType.SQUARE
    assert masts == 2

    # Two-mast fore-and-aft schooner
    arch, rig, masts = extractor.classify_vessel_archetype(
        "Fancy",
        "A fast two-masted schooner built in Baltimore.",
    )
    assert arch == MacroArchetype.TWO_MAST_FORE_AFT
    assert rig == RigType.FORE_AND_AFT
    assert masts == 2

    # One-mast fore-and-aft sloop
    arch, rig, masts = extractor.classify_vessel_archetype(
        "Swift",
        "A Bermuda privateer sloop armed with 8 swivel guns.",
    )
    assert arch == MacroArchetype.ONE_MAST_FORE_AFT
    assert rig == RigType.FORE_AND_AFT
    assert masts == 1


def test_survivorship_bias_and_right_censoring(extractor: WikiMaritimeExtractor):
    """Verify detection of right-censored near-miss casualties (beachings, cargo jettisoning, port arrivals)."""
    # 1. Deliberate beaching (Abraham Wald positive survival observation)
    fate, is_censored = extractor.classify_fate_and_survivorship(
        "The vessel was driven ashore intentionally on the sandbank by her captain to save the crew.",
        ["1715 maritime incidents"],
    )
    assert fate == CasualtyFate.BEACHED_INTENTIONAL
    assert is_censored is True

    # 2. Survived damaged / jettisoned cargo
    fate, is_censored = extractor.classify_fate_and_survivorship(
        "After dismasting, the crew jettisoned 40 tons of ballast and the ship made port at Havana.",
        ["Age of sail ships"],
    )
    assert fate == CasualtyFate.SURVIVED_DAMAGED
    assert is_censored is True

    # 3. Catastrophic foundering (observed loss event)
    fate, is_censored = extractor.classify_fate_and_survivorship(
        "The ship foundered in the great hurricane with the loss of all hands.",
        ["Shipwrecks of the Caribbean"],
    )
    assert fate == CasualtyFate.FOUNDERED_DEEP
    assert is_censored is False


def test_taphonomic_exposure_classification(extractor: WikiMaritimeExtractor):
    """Verify categorization into sheltered_deep, sheltered_topographic, and exposed_high_energy."""
    # Deep sheltered site (> 60m depth, zero bottom wave energy)
    exp = extractor.classify_taphonomic_exposure(
        water_depth_m=4600.0,
        text="Resting on the abyssal plain.",
        lat=23.59,
        lon=-73.89,
    )
    assert exp == TaphonomicExposure.SHELTERED_DEEP

    # Topographic sheltered site (enclosed harbor)
    exp = extractor.classify_taphonomic_exposure(
        water_depth_m=12.0,
        text="Wrecked inside Havana harbor during the storm.",
        lat=23.13,
        lon=-82.35,
    )
    assert exp == TaphonomicExposure.SHELTERED_TOPOGRAPHIC

    # Shallow exposed barrier reef (< 25m depth, high wave energy)
    exp = extractor.classify_taphonomic_exposure(
        water_depth_m=8.0,
        text="Wrecked on the outer Quicksands reef.",
        lat=24.55,
        lon=-82.28,
    )
    assert exp == TaphonomicExposure.EXPOSED_HIGH_ENERGY


def test_stratified_train_test_split():
    """Verify that train/test partitioning preserves macro-archetype and taphonomic proportions."""
    candidates = []
    # Create 20 synthetic candidates with mixed archetypes and exposure classes
    for i in range(20):
        arch = MacroArchetype.THREE_MAST_SQUARE if i % 2 == 0 else MacroArchetype.ONE_MAST_FORE_AFT
        exp = TaphonomicExposure.EXPOSED_HIGH_ENERGY if i % 3 == 0 else TaphonomicExposure.SHELTERED_DEEP
        candidates.append(
            ExtractedWreckCandidate(
                title=f"Wreck_{i}",
                year=1700 + i,
                macro_archetype=arch,
                rig_type=RigType.SQUARE,
                mast_count=3,
                lat=24.0 + (i * 0.1),
                lon=-82.0 - (i * 0.1),
                water_depth_m=20.0,
                casualty_fate=CasualtyFate.GROUNDED_REEF,
                is_censored_survival=False,
                taphonomic_exposure=exp,
                tonnage=500.0,
                armament_guns=30,
                nationality="Spanish",
                extract_snippet="Test extract",
            )
        )

    train, test = WikiMaritimeExtractor.partition_train_test(candidates, train_ratio=0.75, seed=42)
    assert len(train) + len(test) == 20
    assert 13 <= len(train) <= 16
    assert 4 <= len(test) <= 7

    # Ensure both archetypes exist in train and test
    train_archetypes = {c.macro_archetype for c in train}
    test_archetypes = {c.macro_archetype for c in test}
    assert MacroArchetype.THREE_MAST_SQUARE in train_archetypes
    assert MacroArchetype.ONE_MAST_FORE_AFT in train_archetypes
    assert MacroArchetype.THREE_MAST_SQUARE in test_archetypes
    assert MacroArchetype.ONE_MAST_FORE_AFT in test_archetypes

    assert all(c.partition == "train" for c in train)
    assert all(c.partition == "test" for c in test)


def test_archival_provenance_and_versioning(extractor: WikiMaritimeExtractor):
    """Verify archival provenance, Wikipedia canonical URL, dump versioning, and GeoJSON properties."""
    raw_article = {
        "title": "Nuestra Señora de Atocha",
        "extract": "The Spanish galleon sank at 24°33′N 82°17′W during a hurricane in 1622.",
        "categories": ["1622 shipwrecks", "Spanish treasure fleet", "Shipwrecks of Florida"],
    }
    cand = extractor.filter_article(
        raw_article,
        line_idx=1042,
        dump_name="enwiki_13.xml.bz2",
        dump_source="enwiki-2026-06-01",
        dump_year=2026,
    )
    assert cand is not None
    assert cand.title == "Nuestra Señora de Atocha"
    assert cand.wiki_url == "https://en.wikipedia.org/wiki/Nuestra_Se%C3%B1ora_de_Atocha"
    assert cand.dump_source == "enwiki-2026-06-01"
    assert cand.dump_filename == "enwiki_13.xml.bz2"
    assert cand.dump_year == 2026
    assert cand.archive_line == 1042
    assert "nuestra_señora_de_atocha.json" in cand.local_path

    # Verify GeoJSON serialization contains provenance properties
    feat = cand.to_geojson_feature()
    assert feat is not None
    props = feat["properties"]
    assert props["article_name"] == "Nuestra Señora de Atocha"
    assert props["wiki_url"] == cand.wiki_url
    assert props["dump_source"] == "enwiki-2026-06-01"
    assert props["dump_filename"] == "enwiki_13.xml.bz2"
    assert props["dump_year"] == 2026
    assert props["archive_line"] == 1042
    assert props["local_path"] == cand.local_path


def test_article_cache_export_with_full_text(tmp_path: Path, extractor: WikiMaritimeExtractor):
    """Verify that complete raw wikitext articles are exported to individual JSON documents."""
    full_text_sample = "{{Infobox ship | name = Atocha }} The Spanish galleon Nuestra Señora de Atocha sank in the Florida Keys in 1622 with cargo."
    raw_article = {
        "title": "Nuestra Señora de Atocha",
        "extract": full_text_sample,
        "full_text": full_text_sample,
        "categories": ["1622 shipwrecks", "Spanish treasure fleet", "Shipwrecks of Florida"],
    }
    cand = extractor.filter_article(
        raw_article,
        line_idx=100,
        dump_name="enwiki_1.xml.bz2",
    )
    assert cand is not None
    assert cand.full_text == full_text_sample

    # Export to temporary directory
    extractor.export_article_cache([cand], tmp_path)
    article_json = tmp_path / "raw" / "articles" / "nuestra_señora_de_atocha.json"
    assert article_json.exists()

    with open(article_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["title"] == "Nuestra Señora de Atocha"
    assert data["full_text"] == full_text_sample
    assert data["wiki_url"] == "https://en.wikipedia.org/wiki/Nuestra_Se%C3%B1ora_de_Atocha"
