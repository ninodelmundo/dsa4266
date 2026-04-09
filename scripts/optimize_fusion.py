#!/usr/bin/env python3
"""
Run Optuna-based optimization for the fast multimodal fusion model.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")

from src.experiments.common import (
    apply_overrides,
    clone_config,
    flatten_metrics,
    load_fast_resources,
    prepare_environment,
    save_records,
    save_summary,
    train_and_evaluate_model,
)
from src.experiments.tuning import apply_fusion_search_space, build_fusion_model


def build_best_fusion_overrides(promoted_overrides: dict, best_params: dict) -> dict:
    overrides = clone_config(promoted_overrides) if promoted_overrides else {}
    overrides.setdefault("training", {})
    overrides.setdefault("fusion", {})

    for key in [
        "batch_size",
        "optimizer",
        "scheduler",
        "sampling_strategy",
        "class_weights",
        "learning_rate",
        "weight_decay",
    ]:
        if key in best_params:
            overrides["training"][key] = best_params[key]

    if "fusion_dropout" in best_params:
        overrides["fusion"]["dropout"] = best_params["fusion_dropout"]
    for key in ["strategy", "projected_dim", "hidden_dim", "dropout"]:
        if key in best_params:
            overrides["fusion"][key] = best_params[key]

    overrides["training"]["checkpoint_metric"] = "composite_score"
    return overrides


def main():
    parser = argparse.ArgumentParser(description="Optimize fast multimodal fusion with Optuna.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file.")
    parser.add_argument("--trials", type=int, default=None, help="Override the configured number of trials.")
    parser.add_argument(
        "--promoted-overrides",
        default=None,
        help="Path to promoted unimodal overrides JSON. Defaults to outputs/optimization/unimodal/promoted_overrides.json if present.",
    )
    args = parser.parse_args()

    import optuna

    config, device, logger = prepare_environment(args.config)
    if args.trials is not None:
        config["optimization"]["n_trials"] = args.trials

    default_promoted_path = Path(config["project"]["output_dir"]) / "optimization" / "unimodal" / "promoted_overrides.json"
    promoted_path = Path(args.promoted_overrides) if args.promoted_overrides else default_promoted_path
    promoted_overrides = {}
    if promoted_path.exists():
        promoted_overrides = json.loads(promoted_path.read_text())
        apply_overrides(config, promoted_overrides)
        logger.info("Loaded promoted unimodal overrides from %s", promoted_path)

    df, features = load_fast_resources(config)
    output_root = Path(config["project"]["output_dir"]) / "optimization" / "fusion"
    output_root.mkdir(parents=True, exist_ok=True)

    records = []
    best_results = {}

    def objective(trial):
        trial_config = clone_config(config)
        selected_params = apply_fusion_search_space(trial_config, trial)
        trial_dir = output_root / f"trial_{trial.number:03d}"

        try:
            model, model_type, model_name = build_fusion_model(trial_config)
            results = train_and_evaluate_model(
                model=model,
                model_type=model_type,
                model_name=model_name,
                config=trial_config,
                device=device,
                df=df,
                features=features,
                output_dir=str(trial_dir),
                trial=trial,
            )
        except optuna.TrialPruned:
            records.append(
                {
                    "trial_number": trial.number,
                    "status": "pruned",
                    **selected_params,
                }
            )
            raise

        record = {
            "trial_number": trial.number,
            "status": "completed",
            **selected_params,
            **flatten_metrics("val", results["val_metrics"]),
            **flatten_metrics("test", results["test_metrics"]),
            "checkpoint_path": results["checkpoint_path"],
            "optimal_threshold": results["optimal_threshold"],
            "calibration_temperature": results["calibration_temperature"],
        }
        records.append(record)
        best_results[trial.number] = {
            "params": selected_params,
            "record": record,
            "results": results,
        }
        return results["val_metrics"]["composite_score"]

    pruner = None
    if config.get("runtime", {}).get("prune_trials", True):
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)

    study = optuna.create_study(
        direction="maximize",
        study_name="fusion_optimization",
        pruner=pruner,
    )
    study.optimize(objective, n_trials=config["optimization"]["n_trials"])

    save_records(records, str(output_root), "trial_results")
    best_trial_info = best_results.get(study.best_trial.number, {})
    best_overrides = build_best_fusion_overrides(
        promoted_overrides,
        best_trial_info.get("params", {}),
    )
    save_summary(best_overrides, str(output_root), "best_fusion_overrides.json")
    save_summary(
        {
            "best_value": study.best_value,
            "best_params": best_trial_info.get("params", {}),
            "best_record": best_trial_info.get("record", {}),
            "n_trials": len(study.trials),
            "promoted_overrides_path": str(promoted_path) if promoted_path.exists() else None,
        },
        str(output_root),
        "summary.json",
    )
    logger.info("Saved fusion optimization artifacts to %s", output_root)


if __name__ == "__main__":
    main()
