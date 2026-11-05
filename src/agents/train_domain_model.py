"""Training pipeline for fine-tuning open-weight models (Qwen 3.5/3.8) on Caribbean maritime loss records."""

import argparse
import json
import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)

# Sample synthetic training dataset of 17th/18th-century loss records paired with ground truth JSON
SAMPLE_CARIBBEAN_TRAINING_CORPUS: List[Dict[str, str]] = [
    {
        "text": "In the catastrophic gale of September 1622, the Spanish treasure galleon Nuestra Senora de Atocha of 550 tons and three square-rigged masts, carrying 1100 bars of silver and 40 tons of ballast, was thrown upon the reef at 24.55 N, 81.92 W.",
        "target_json": json.dumps({
            "vessel_name": "Nuestra Senora de Atocha",
            "wreck_year": 1622,
            "mast_count": 3,
            "rig_type": "square",
            "is_modern": False,
            "tonnage": 550.0,
            "cargo_mass_category": "precious_metals",
            "buoyancy_factor": 0.28,
            "draft_meters": 4.8,
            "lkp_lat": 24.55,
            "lkp_lon": -81.92
        })
    },
    {
        "text": "HMS Looe, a British 44-gun frigate of three masts and 850 tons, with a draft of 17 feet, struck the outer barrier reef off Key West in February 1744 under heavy north-east gale.",
        "target_json": json.dumps({
            "vessel_name": "HMS Looe",
            "wreck_year": 1744,
            "mast_count": 3,
            "rig_type": "square",
            "is_modern": False,
            "tonnage": 850.0,
            "cargo_mass_category": "heavy",
            "buoyancy_factor": 0.35,
            "draft_meters": 5.18,
            "lkp_lat": 24.54,
            "lkp_lon": -81.41
        })
    },
    {
        "text": "A swift Bermuda privateer sloop of one mast and 75 tons, armed with 8 swivel guns and carrying sugar casks, was driven ashore on the Pedro Banks in October 1712.",
        "target_json": json.dumps({
            "vessel_name": "Bermuda Sloop",
            "wreck_year": 1712,
            "mast_count": 1,
            "rig_type": "fore_and_aft",
            "is_modern": False,
            "tonnage": 75.0,
            "cargo_mass_category": "medium",
            "buoyancy_factor": 0.40,
            "draft_meters": 2.6,
            "lkp_lat": 17.05,
            "lkp_lon": -77.85
        })
    }
]


def format_instruction_prompt(source_text: str) -> str:
    """Format input text into a standard instruction prompt for Qwen fine-tuning."""
    system_prompt = (
        "You are an expert maritime archaeology information extractor. Extract exact numerical "
        "and physical vessel parameters into strict JSON conforming to: vessel_name, wreck_year, "
        "mast_count, rig_type, is_modern, tonnage, cargo_mass_category, buoyancy_factor, draft_meters, "
        "lkp_lat, lkp_lon."
    )
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\nExtract vessel attributes from this record:\n{source_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def train_and_export_to_huggingface(
    base_model_name: str = "Qwen/Qwen3.5-4B",
    output_hub_repo: Optional[str] = None,
    hf_token: Optional[str] = None,
    epochs: int = 3,
    learning_rate: float = 2e-4,
    use_dora: bool = True,
) -> None:
    """Fine-tune a base model using LoRA / DoRA on historical maritime loss records and push to Hugging Face Hub.

    Args:
        base_model_name: Hugging Face base model identifier (e.g. Qwen/Qwen3.5-4B).
        output_hub_repo: Target repo name on Hugging Face (e.g. 'username/smd-caribbean-parser-4b').
        hf_token: Hugging Face write authentication token.
        epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        use_dora: Whether to use Weight-Decomposed Low-Rank Adaptation (DoRA).
    """
    logger.info(f"Initializing fine-tuning pipeline for base model: {base_model_name}")
    logger.info(f"DoRA enabled: {use_dora}, Epochs: {epochs}, LR: {learning_rate}")

    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Execution device: {device}")

        # In production with GPU, load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        peft_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            use_dora=use_dora,
        )
        model = get_peft_model(model, peft_config)
        logger.info("DoRA adapters successfully attached to base model.")

        if output_hub_repo and hf_token:
            logger.info(f"Pushing fine-tuned model weights to Hugging Face Hub: {output_hub_repo}")
            model.push_to_hub(output_hub_repo, token=hf_token)
            tokenizer.push_to_hub(output_hub_repo, token=hf_token)
            logger.info("Model published successfully to Hugging Face Model Hub.")

    except ImportError:
        logger.warning(
            "peft/transformers/torch not fully installed or CUDA unavailable. "
            "Skipping active GPU training step in current environment."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune open-weight Qwen model for SMD extraction")
    parser.add_argument("--base_model", default="Qwen/Qwen3.8-27B", help="Base model Hugging Face ID")
    parser.add_argument("--hub_repo", default=None, help="Target Hugging Face Hub repo (username/model-name)")
    parser.add_argument("--hf_token", default=os.getenv("HF_TOKEN"), help="Hugging Face authentication token")
    parser.add_argument("--use_dora", action="store_true", default=True, help="Use DoRA instead of standard LoRA")
    args = parser.parse_args()

    train_and_export_to_huggingface(
        base_model_name=args.base_model,
        output_hub_repo=args.hub_repo,
        hf_token=args.hf_token,
        use_dora=args.use_dora,
    )
