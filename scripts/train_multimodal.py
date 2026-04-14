#!/usr/bin/env python3
"""
Train the baseline multi-modal fusion model using the default configuration
in configs/config.yaml.

Mirrors the output layout of scripts/optimize_fusion.py so baseline and
Optuna-tuned runs live under a parallel directory tree and share the same
summary.json schema:

    outputs/baseline/fusion/
        best_model.pt
        training_curves_multimodal-fusion.png
        trial_results.{csv,json}
        summary.json
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")

from src.experiments.common import (
    clone_config,
    flatten_metrics,
    load_fast_resources,
    prepare_environment,
    save_records,
    save_summary,
    train_and_evaluate_model,
)
from src.experiments.tuning import build_fusion_model


def _extract_fusion_hyperparams(config: dict) -> dict:
    """Snapshot of the config values that the fusion Optuna search records,
    so baseline and tuned summaries share the same best_params schema."""
    return {
        "batch_size": config["training"]["batch_size"],
        "optimizer": config["training"]["optimizer"],
        "scheduler": config["training"]["scheduler"],
        "sampling_strategy": config["training"]["sampling_strategy"],
        "class_weights": config["training"]["class_weights"],
        "learning_rate": config["training"]["learning_rate"],
        "weight_decay": config["training"]["weight_decay"],
        "fusion_dropout": config["fusion"]["dropout"],
        "strategy": config["fusion"]["strategy"],
        "projected_dim": config["fusion"]["projected_dim"],
        "hidden_dim": config["fusion"]["hidden_dim"],
        "dropout": config["fusion"]["dropout"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train the baseline fusion model with the default config.",
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file.")
    args = parser.parse_args()

    config, device, logger = prepare_environment(args.config)
    df, features = load_fast_resources(config)

    output_dir = Path(config["project"]["output_dir"]) / "baseline" / "fusion"
    output_dir.mkdir(parents=True, exist_ok=True)

    trial_config = clone_config(config)
    model, model_type, model_name = build_fusion_model(trial_config)
    logger.info(
        "Training baseline fusion model (strategy=%s, projected_dim=%d)",
        trial_config["fusion"]["strategy"],
        trial_config["fusion"]["projected_dim"],
    )
    results = train_and_evaluate_model(
        model=model,
        model_type=model_type,
        model_name=model_name,
        config=trial_config,
        device=device,
        df=df,
        features=features,
        output_dir=str(output_dir),
    )

    selected_params = _extract_fusion_hyperparams(trial_config)
    record = {
        "trial_number": 0,
        "status": "completed",
        **selected_params,
        **flatten_metrics("val", results["val_metrics"]),
        **flatten_metrics("test", results["test_metrics"]),
        "checkpoint_path": results["checkpoint_path"],
        "optimal_threshold": results["optimal_threshold"],
        "calibration_temperature": results["calibration_temperature"],
    }
    save_records([record], str(output_dir), "trial_results")

    summary = {
        "model_name": model_name,
        "config_source": "baseline",
        "n_trials": 1,
        "best_value": float(results["val_metrics"]["composite_score"]),
        "best_params": selected_params,
        "best_record": record,
    }
    if results.get("active_modalities") is not None:
        summary["active_modalities"] = results["active_modalities"]
    if results.get("modality_weights") is not None:
        summary["modality_weights"] = results["modality_weights"]

    save_summary(summary, str(output_dir), "summary.json")
    logger.info("Saved baseline fusion artifacts to %s", output_dir)


if __name__ == "__main__":
    main()
