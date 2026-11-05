"""Archival text extraction engine connecting to host URLs (Hugging Face / vLLM / Ollama) with deterministic fallback."""

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple
import requests

from src.agents.judge import evaluate_extraction_quality
from src.agents.schemas import JudgeEvaluationRecord, VesselExtractionRecord
from src.config import settings
from src.profiles.vessel_class import RigType

logger = logging.getLogger(__name__)


def heuristic_text_parser(raw_text: str) -> VesselExtractionRecord:
    """Extract vessel attributes from raw text using regex patterns and nautical vocabularies.

    Args:
        raw_text: Historical narrative or registry entry.

    Returns:
        Populated VesselExtractionRecord.
    """
    text_lower = raw_text.lower()

    # 1. Vessel Name Extraction
    name_match = re.search(
        r"(?:hms|ship|vessel|galleon|sloop|brig|nao)\s+([A-Z][a-zA-Z\s'\-]{2,25})",
        raw_text,
    )
    if name_match:
        vessel_name = name_match.group(0).strip()
    else:
        # Check first line or quotes
        quote_match = re.search(r"\"([A-Za-z\s'\-]{3,30})\"", raw_text)
        vessel_name = quote_match.group(1).strip() if quote_match else "Historical Vessel"

    # 2. Wreck Year
    year_match = re.search(r"\b(1[5-9]\d{2}|20[0-2]\d)\b", raw_text)
    wreck_year = int(year_match.group(1)) if year_match else None

    # 3. Modern vs Wooden Hull
    modern_keywords = ["container", "diesel", "steam", "steel", "bulk carrier", "cargo ship", "tanker"]
    is_modern = any(k in text_lower for k in modern_keywords)

    # 4. Mast Count and Rig Type
    mast_count = 3  # Default
    rig_type = RigType.SQUARE

    if is_modern:
        mast_count = 0
        rig_type = RigType.UNMASTED_POWER
    elif "one mast" in text_lower or "single mast" in text_lower or "sloop" in text_lower or "cutter" in text_lower:
        mast_count = 1
        rig_type = RigType.FORE_AND_AFT
    elif "schooner" in text_lower or "ketch" in text_lower:
        mast_count = 2
        rig_type = RigType.FORE_AND_AFT
    elif "brig" in text_lower or "snow" in text_lower or "brigantine" in text_lower:
        mast_count = 2
        rig_type = RigType.SQUARE
    elif any(k in text_lower for k in ["galleon", "frigate", "nao", "fluyt", "carrack", "three mast"]):
        mast_count = 3
        rig_type = RigType.SQUARE

    # 5. Tonnage
    tonnage = None
    ton_match = re.search(r"(\d{2,5})\s*(?:tons|ton|toneladas|burden)", text_lower)
    if ton_match:
        tonnage = float(ton_match.group(1))

    # 6. Draft
    draft_meters = None
    draft_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:meters|m|feet|ft)\s*(?:draft|draught)", text_lower)
    if draft_match:
        val = float(draft_match.group(1))
        # If stated in feet, convert to meters
        if "foot" in text_lower or "feet" in text_lower or "ft" in text_lower:
            draft_meters = round(val * 0.3048, 2)
        else:
            draft_meters = round(val, 2)

    # 7. Cargo Category
    if any(k in text_lower for k in ["silver", "gold", "coins", "treasure", "specie", "reales"]):
        cargo_cat = "precious_metals"
        buoyancy = 0.28
    elif any(k in text_lower for k in ["iron", "cannon", "ballast", "lead", "ordnance", "heavy"]):
        cargo_cat = "heavy"
        buoyancy = 0.32
    elif any(k in text_lower for k in ["tobacco", "sugar", "coffee", "cotton", "timber", "cacao"]):
        cargo_cat = "medium"
        buoyancy = 0.40
    else:
        cargo_cat = "medium"
        buoyancy = 0.35

    # 8. Coordinates
    lat, lon = None, None
    coord_match = re.search(
        r"(\d{1,2}(?:\.\d+)?)\s*°?\s*([NS])\s*[,;\s]+\s*(\d{1,3}(?:\.\d+)?)\s*°?\s*([EW])",
        raw_text,
        re.IGNORECASE,
    )
    if coord_match:
        lat_val = float(coord_match.group(1))
        lat_hem = coord_match.group(2).upper()
        lon_val = float(coord_match.group(3))
        lon_hem = coord_match.group(4).upper()

        lat = lat_val if lat_hem == "N" else -lat_val
        lon = lon_val if lon_hem == "E" else -lon_val

    return VesselExtractionRecord(
        vessel_name=vessel_name,
        wreck_year=wreck_year,
        mast_count=mast_count,
        rig_type=rig_type,
        is_modern=is_modern,
        tonnage=tonnage,
        cargo_mass_category=cargo_cat,
        buoyancy_factor=buoyancy,
        draft_meters=draft_meters,
        lkp_lat=lat,
        lkp_lon=lon,
        loss_narrative=raw_text.strip(),
    )


def query_llm_host(
    prompt: str,
    endpoint_url: str,
    model_name: str,
    api_key: Optional[str] = None,
    timeout_s: float = 30.0,
) -> Optional[Dict[str, Any]]:
    """Query an OpenAI-compatible /v1/chat/completions endpoint (vLLM, Ollama, Hugging Face).

    Args:
        prompt: User text prompt.
        endpoint_url: URL to OpenAI-compatible base (e.g. http://localhost:8000/v1).
        model_name: Deployed model identifier.
        api_key: Bearer token if endpoint requires authentication.
        timeout_s: Network timeout in seconds.

    Returns:
        Parsed JSON dictionary response if successful, None otherwise.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    system_instruction = (
        "You are an expert maritime archaeologist and naval architecture extractor. "
        "Extract structured JSON adhering strictly to: vessel_name (str), wreck_year (int or null), "
        "mast_count (int: 0, 1, 2, or 3), rig_type ('square', 'fore_and_aft', 'unmasted_power'), "
        "is_modern (bool), tonnage (float or null), cargo_mass_category ('light', 'medium', 'heavy', 'precious_metals'), "
        "buoyancy_factor (float 0.1-0.8), draft_meters (float or null), lkp_lat (float or null), lkp_lon (float or null). "
        "Return ONLY raw JSON."
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
    }

    try:
        url = f"{endpoint_url.rstrip('/')}/chat/completions"
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)
    except Exception as exc:
        logger.debug(f"LLM host query failed ({exc}); falling back to heuristic parser.")
        return None

    return None


def extract_vessel_record(
    raw_text: str,
    use_remote_host: bool = True,
    validate_with_judge: bool = True,
) -> Tuple[VesselExtractionRecord, JudgeEvaluationRecord]:
    """Parse raw historical text into a validated VesselExtractionRecord and Judge evaluation.

    Attempts remote inference at settings.llm_host_url first; falls back gracefully to
    the heuristic rule engine if offline or unreachable.

    Args:
        raw_text: Unstructured historical text or registry summary.
        use_remote_host: Whether to attempt remote LLM API call.
        validate_with_judge: Whether to run Kev/Jev verification on the extracted record.

    Returns:
        Tuple of (VesselExtractionRecord, JudgeEvaluationRecord).
    """
    record: Optional[VesselExtractionRecord] = None

    if use_remote_host:
        json_data = query_llm_host(
            prompt=raw_text,
            endpoint_url=settings.llm_host_url,
            model_name=settings.parser_model_name,
            api_key=settings.llm_api_key,
            timeout_s=settings.llm_timeout_seconds,
        )
        if json_data:
            try:
                # Merge narrative back
                json_data["loss_narrative"] = raw_text.strip()
                record = VesselExtractionRecord(**json_data)
            except Exception as e:
                logger.debug(f"Failed to validate LLM response schema: {e}")

    # Fallback to local heuristic extractor if remote call returned None or failed
    if record is None:
        record = heuristic_text_parser(raw_text)

    # Evaluate fidelity and credibility using Kev / Jev
    judge_eval = (
        evaluate_extraction_quality(raw_text, record, use_remote_judge=use_remote_host)
        if validate_with_judge
        else JudgeEvaluationRecord(evaluator_name="unvalidated")
    )

    return record, judge_eval
