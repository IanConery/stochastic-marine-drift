"""System-One decision judge integrating Kev (open-source) and Jev (TypeSafe) protocols."""

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple
import requests

from src.agents.schemas import JudgeEvaluationRecord, VesselExtractionRecord
from src.config import settings

logger = logging.getLogger(__name__)


def heuristic_judge_evaluation(
    raw_text: str, record: VesselExtractionRecord
) -> JudgeEvaluationRecord:
    """Deterministic fallback evaluator applying linguistic and physical consistency rules.

    Analyzes raw text for archival provenance keywords, mast count contradictions,
    and physical plausibility to simulate the System-One discriminative head.

    Args:
        raw_text: Original historical narrative text.
        record: Extracted VesselExtractionRecord.

    Returns:
        JudgeEvaluationRecord instance.
    """
    text_lower = raw_text.lower()
    hallucination = False
    notes = []

    # 1. Mast count verification
    # Detect explicit mentions of mast count in text
    if "one mast" in text_lower or "single mast" in text_lower or "single-masted" in text_lower:
        if record.mast_count != 1:
            hallucination = True
            notes.append(f"Text mentions single mast, but record has mast_count={record.mast_count}")
    elif "two mast" in text_lower or "two-masted" in text_lower:
        if record.mast_count != 2:
            hallucination = True
            notes.append(f"Text mentions two masts, but record has mast_count={record.mast_count}")
    elif "three mast" in text_lower or "three-masted" in text_lower:
        if record.mast_count != 3:
            hallucination = True
            notes.append(f"Text mentions three masts, but record has mast_count={record.mast_count}")

    # 2. Modern vs Wooden consistency
    if any(k in text_lower for k in ["container", "diesel", "steamship", "steel hull", "bulk carrier"]):
        if not record.is_modern:
            hallucination = True
            notes.append("Modern industrial terms present, but is_modern is False")
    elif any(k in text_lower for k in ["galleon", "flota", "spanish plate", "frigate", "wood", "rigging"]):
        if record.is_modern:
            hallucination = True
            notes.append("Historical timber terms present, but is_modern is True")

    # 3. Provenance Tier Classification
    tier = 2  # Default to convoy sighting
    tier_confidence = 0.85

    # Tier 1 indicators (Direct eyewitness captain/officer journal)
    t1_patterns = [
        r"\bi\b", r"\bmy journal\b", r"\bour ship\b", r"\bwe struck\b",
        r"\bcaptain's log\b", r"\bat \d+ bells\b", r"\bour rudder\b",
        r"\bwe cast anchor\b", r"\bwe cut away our mast\b"
    ]
    if any(re.search(pat, text_lower) for pat in t1_patterns):
        tier = 1
        tier_confidence = 0.95
    # Tier 3 indicators (Retrospective colonial inquest / rumor)
    t3_patterns = [
        r"\breported in havana\b", r"\brumor\b", r"\bletter to the governor\b",
        r"\bsome months later\b", r"\bit is believed\b", r"\bsupposed to have foundered\b",
        r"\bnews reached\b", r"\binquest\b"
    ]
    if any(re.search(pat, text_lower) for pat in t3_patterns):
        tier = 3
        tier_confidence = 0.88

    is_factual = not hallucination
    confidence = tier_confidence if is_factual else 0.40

    return JudgeEvaluationRecord(
        is_factual=is_factual,
        confidence_score=confidence,
        assigned_tier=tier,
        hallucination_detected=hallucination,
        evaluator_name="heuristic_rule_engine",
        notes="; ".join(notes) if notes else "Extraction physically and textually consistent",
    )


def query_systemone_endpoint(
    raw_text: str,
    record: VesselExtractionRecord,
    endpoint_url: str,
    model_name: str,
    api_key: Optional[str] = None,
    timeout_s: float = 10.0,
) -> Optional[JudgeEvaluationRecord]:
    """Query a Kev or Jev System-One HTTP endpoint for discriminative evaluation.

    Args:
        raw_text: Source historical text.
        record: Extracted candidate record.
        endpoint_url: Base URL (e.g., http://localhost:8000/v1/systemone).
        model_name: Model identifier (e.g. 'jaredpalmer/kev-0.8b' or 'typesafe/jev-latest').
        api_key: Optional bearer token or API key.
        timeout_s: Network timeout in seconds.

    Returns:
        JudgeEvaluationRecord if successful, None if network call fails.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_name,
        "state": {
            "source_text": raw_text,
            "extracted_json": record.model_dump(),
        },
        "questions": {
            "is_factual": {
                "type": "noul",
                "question": "Are all extracted parameters (mast count, rig type, draft, circumstances) faithful to the source text without hallucination?",
            },
            "credibility_tier": {
                "type": "choice",
                "question": "What is the archival source credibility tier of this text?",
                "choices": [1, 2, 3],
            },
        },
    }

    try:
        url = f"{endpoint_url.rstrip('/')}/systemone"
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
        if resp.status_code == 200:
            data = resp.json()
            is_factual = bool(data.get("decisions", {}).get("is_factual", {}).get("value", True))
            confidence = float(data.get("decisions", {}).get("is_factual", {}).get("probability", 0.90))
            tier = int(data.get("decisions", {}).get("credibility_tier", {}).get("value", 2))

            return JudgeEvaluationRecord(
                is_factual=is_factual,
                confidence_score=confidence,
                assigned_tier=tier if tier in [1, 2, 3] else 2,
                hallucination_detected=not is_factual,
                evaluator_name=model_name,
                notes="Evaluated via Kev/Jev System-One endpoint",
            )
    except Exception as exc:
        logger.debug(f"System-One endpoint call failed ({exc}); falling back to heuristic judge.")
        return None

    return None


def evaluate_extraction_quality(
    raw_text: str,
    record: VesselExtractionRecord,
    use_remote_judge: bool = True,
) -> JudgeEvaluationRecord:
    """Evaluate extraction fidelity and source credibility using Kev/Jev with fallback.

    Args:
        raw_text: Source historical text.
        record: Extracted VesselExtractionRecord.
        use_remote_judge: Whether to attempt remote Kev/Jev endpoint call.

    Returns:
        JudgeEvaluationRecord.
    """
    if use_remote_judge and settings.judge_enabled:
        result = query_systemone_endpoint(
            raw_text=raw_text,
            record=record,
            endpoint_url=settings.judge_host_url,
            model_name=settings.judge_model_name,
            api_key=settings.typesafe_api_key or settings.llm_api_key,
            timeout_s=settings.llm_timeout_seconds,
        )
        if result is not None:
            return result

    # Default to robust local heuristic engine
    return heuristic_judge_evaluation(raw_text, record)
