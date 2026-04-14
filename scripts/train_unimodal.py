#!/usr/bin/env python3
"""
Train baseline unimodal models (URL, Text, Visual, HTML) using the default
configuration in configs/config.yaml.

Mirrors the output layout of scripts/optimize_unimodal.py so baseline and
Optuna-tuned results live under a matching directory tree and share the same
summary.json schema:

    outputs/baseline/unimodal/{url,text,visual,html}/
        best_model.pt
        training_curves_<model-name>.png
        trial_results.{csv,json}
        summary.json
    outputs/baseline/unimodal/summary.json          (aggregate)

Each per-modality summary.json matches the Optuna schema (see
scripts/optimize_unimodal.py) so downstream consumers (evaluate.py,
run_ablation.py, run_explainability.py) can read baseline and optimised
runs through the same helpers.
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
from src.experiments.tuning import build_unimodal_model


def _extract_modality_hyperparams(config: dict, modality: str) -> dict:
    """Snapshot of the config values that unimodal Optuna trials also record,
    so baseline and tuned summaries share the same best_params schema."""
    params = {
        "batch_size": config["training"]["batch_size"],
        "optimizer": config["training"]["optimizer"],
        "scheduler": config["training"]["scheduler"],
        "sampling_strategy": config["training"]["sampling_strategy"],
        "class_weights": config["training"]["class_weights"],
        "learning_rate": config["training"]["learning_rate"],
        "weight_decay": config["training"]["weight_decay"],
        "fusion_dropout": config["fusion"]["dropout"],
    }
    if modality == "url":
        url_cfg = config["url"]
        params.update({
            "model_type": url_cfg["model_type"],
            "embedding_dim": url_cfg["embedding_dim"],
            "hidden_dim": url_cfg["hidden_dim"],
            "dropout": url_cfg["dropout"],
            "classifier_hidden_dim": url_cfg["classifier_hidden_dim"],
            "classifier_bottleneck_dim": url_cfg["classifier_bottleneck_dim"],
            "use_url_scalar_features": url_cfg["use_url_scalar_features"],
        })
    elif modality == "text":
        text_cfg = config["text"]
        params.update({
            "dropout": text_cfg["dropout"],
            "classifier_hidden_dim": text_cfg["classifier_hidden_dim"],
            "classifier_bottleneck_dim": text_cfg["classifier_bottleneck_dim"],
        })
    elif modality == "visual":
        visual_cfg = config["visual"]
        params.update({
            "dropout": visual_cfg["dropout"],
            "classifier_hidden_dim": visual_cfg["classifier_hidden_dim"],
            "classifier_bottleneck_dim": visual_cfg["classifier_bottleneck_dim"],
        })
    elif modality == "html":
        html_cfg = config["html"]
        params.update({
            "classifier_hidden_dim": html_cfg["classifier_hidden_dim"],
            "classifier_bottleneck_dim": html_cfg["classifier_bottleneck_dim"],
        })
    return params


def train_baseline_modality(modality: str, base_config: dict, device, df, features, root_dir: Path):
    trial_config = clone_config(base_config)
    modality_dir = root_dir / modality
    modality_dir.mkdir(parents=True, exist_ok=True)

    model, model_type, model_name = build_unimodal_model(modality, trial_config)
    results = train_and_evaluate_model(
        model=model,
        model_type=model_type,
        model_name=model_name,
        config=trial_config,
        device=device,
        df=df,
        features=features,
        output_dir=str(modality_dir),
    )

    selected_params = _extract_modality_hyperparams(trial_config, modality)
    record = {
        "trial_number": 0,
        "modality": modality,
        "status": "completed",
        **selected_params,
        **flatten_metrics("val", results["val_metrics"]),
        **flatten_metrics("test", results["test_metrics"]),
        "checkpoint_path": results["checkpoint_path"],
        "optimal_threshold": results["optimal_threshold"],
        "calibration_temperature": results["calibration_temperature"],
    }
    save_records([record], str(modality_dir), "trial_results")

    summary = {
        "modality": modality,
        "config_source": "baseline",
        "n_trials": 1,
        "best_value": float(results["val_metrics"]["composite_score"]),
        "best_params": selected_params,
        "best_record": record,
    }
    save_summary(summary, str(modality_dir), "summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline unimodal models with the default config.",
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file.")
    parser.add_argument(
        "--modality",
        choices=["url", "text", "visual", "html", "all"],
        default="all",
        help="Which unimodal baseline to train.",
    )
    args = parser.parse_args()

    config, device, logger = prepare_environment(args.config)
    df, features = load_fast_resources(config)

    output_root = Path(config["project"]["output_dir"]) / "baseline" / "unimodal"
    output_root.mkdir(parents=True, exist_ok=True)

    modalities = ["url", "text", "visual", "html"] if args.modality == "all" else [args.modality]
    summaries = {}
    for modality in modalities:
        logger.info("Training baseline unimodal model for %s", modality)
        summaries[modality] = train_baseline_modality(
            modality, config, device, df, features, output_root,
        )

    save_summary(
        {
            "config_source": "baseline",
            "summaries": summaries,
        },
        str(output_root),
        "summary.json",
    )
    logger.info("Saved baseline unimodal artifacts to %s", output_root)


if __name__ == "__main__":
    main()
