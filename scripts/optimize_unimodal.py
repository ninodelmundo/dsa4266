#!/usr/bin/env python3
"""
Run Optuna-based optimization for fast unimodal baselines.
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
from src.experiments.tuning import (
    apply_unimodal_search_space,
    build_promoted_overrides,
    build_unimodal_model,
)


def optimize_modality(modality: str, base_config: dict, device, df, features, root_dir: Path):
    import optuna

    modality_dir = root_dir / modality
    modality_dir.mkdir(parents=True, exist_ok=True)
    records = []
    best_results = {}

    def objective(trial):
        trial_config = clone_config(base_config)
        selected_params = apply_unimodal_search_space(trial_config, trial, modality)
        trial_dir = modality_dir / f"trial_{trial.number:03d}"

        try:
            model, model_type, model_name = build_unimodal_model(modality, trial_config)
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
                    "modality": modality,
                    "status": "pruned",
                    **selected_params,
                }
            )
            raise

        record = {
            "trial_number": trial.number,
            "modality": modality,
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
            "results": results,
            "record": record,
        }
        return results["val_metrics"]["composite_score"]

    pruner = None
    if base_config.get("runtime", {}).get("prune_trials", True):
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{modality}_optimization",
        pruner=pruner,
    )
    study.optimize(objective, n_trials=base_config["optimization"]["n_trials"])

    save_records(records, str(modality_dir), "trial_results")
    best_trial_info = best_results.get(study.best_trial.number, {})
    summary = {
        "modality": modality,
        "best_value": study.best_value,
        "best_params": best_trial_info.get("params", {}),
        "best_record": best_trial_info.get("record", {}),
        "n_trials": len(study.trials),
    }
    save_summary(summary, str(modality_dir), "summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Optimize fast unimodal baselines with Optuna.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--modality",
        choices=["url", "text", "visual", "html", "all"],
        default="all",
        help="Which unimodal study to run.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Override the configured number of trials.",
    )
    args = parser.parse_args()

    import optuna  # noqa: F401  # imported here so --help works without dependencies

    config, device, logger = prepare_environment(args.config)
    if args.trials is not None:
        config["optimization"]["n_trials"] = args.trials

    df, features = load_fast_resources(config)
    output_root = Path(config["project"]["output_dir"]) / "optimization" / "unimodal"
    output_root.mkdir(parents=True, exist_ok=True)

    modalities = ["url", "text", "visual", "html"] if args.modality == "all" else [args.modality]
    summaries = {}
    for modality in modalities:
        logger.info("Starting unimodal optimization for %s", modality)
        summaries[modality] = optimize_modality(
            modality,
            config,
            device,
            df,
            features,
            output_root,
        )

    promoted_overrides = build_promoted_overrides(summaries)
    save_summary(promoted_overrides, str(output_root), "promoted_overrides.json")
    save_summary(summaries, str(output_root), "summary.json")
    logger.info("Saved unimodal optimization artifacts to %s", output_root)


if __name__ == "__main__":
    main()
