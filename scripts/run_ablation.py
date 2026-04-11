#!/usr/bin/env python3
"""
Run systematic ablation analysis for the fast fusion model.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.metrics import compute_composite_score
from src.experiments.common import (
    apply_overrides,
    clone_config,
    flatten_metrics,
    get_metric_weights,
    load_fast_resources,
    parse_logged_baseline_metrics,
    prepare_environment,
    save_records,
    save_summary,
    train_and_evaluate_model,
)
from src.experiments.tuning import build_fusion_model


def build_ablation_variants(config: dict):
    variants = list(config.get("ablation", {}).get("variants", []))
    compare_strategies = config.get("ablation", {}).get("compare_strategies", [])
    for strategy in compare_strategies:
        variants.append(
            {
                "name": f"strategy_{strategy}",
                "disabled_modalities": [],
                "use_url_scalar_features": True,
                "fusion_strategy": strategy,
            }
        )
    return variants


def metric_delta(record: dict, anchor: dict, metric_name: str):
    if anchor is None:
        return float("nan")
    return float(record.get(metric_name, float("nan")) - anchor.get(metric_name, float("nan")))


def plot_ablation_metrics(df: pd.DataFrame, output_dir: Path):
    metrics = ["test_f1", "test_roc_auc", "test_c_index", "test_composite_score"]
    plot_df = df[["variant"] + metrics].set_index("variant")
    ax = plot_df.plot(kind="bar", figsize=(13, 6))
    ax.set_title("Ablation Metrics Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0.6, 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(output_dir / "ablation_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_ablation_deltas(df: pd.DataFrame, output_dir: Path):
    metrics = [
        "delta_vs_full_test_f1",
        "delta_vs_full_test_roc_auc",
        "delta_vs_full_test_c_index",
        "delta_vs_full_test_composite_score",
    ]
    plot_df = df[["variant"] + metrics].set_index("variant")
    ax = plot_df.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Ablation Delta vs Full Tuned Fusion")
    ax.set_ylabel("Delta")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_deltas.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run fast-fusion ablation analysis.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file.")
    parser.add_argument(
        "--fusion-overrides",
        default=None,
        help="Path to best fusion overrides JSON. Defaults to outputs/optimization/fusion/best_fusion_overrides.json if present.",
    )
    args = parser.parse_args()

    config, device, logger = prepare_environment(args.config)
    metric_weights = get_metric_weights(config)

    default_overrides_path = Path(config["project"]["output_dir"]) / "optimization" / "fusion" / "best_fusion_overrides.json"
    overrides_path = Path(args.fusion_overrides) if args.fusion_overrides else default_overrides_path
    if overrides_path.exists():
        apply_overrides(config, json.loads(overrides_path.read_text()))
        logger.info("Loaded fusion overrides from %s", overrides_path)

    config["training"]["checkpoint_metric"] = "composite_score"
    df, features = load_fast_resources(config)
    output_root = Path(config["project"]["output_dir"]) / "optimization" / "ablation"
    output_root.mkdir(parents=True, exist_ok=True)

    baseline_metrics = parse_logged_baseline_metrics(
        str(Path(config["project"]["output_dir"]) / "run.log")
    )
    if baseline_metrics is not None:
        baseline_metrics["c_index"] = baseline_metrics.get("roc_auc", float("nan"))
        baseline_metrics["composite_score"] = compute_composite_score(
            baseline_metrics,
            metric_weights,
        )

    variants = build_ablation_variants(config)
    records = []
    full_reference = None

    for variant in variants:
        variant_name = variant["name"]
        variant_config = clone_config(config)
        variant_config["fusion"]["disabled_modalities"] = variant.get("disabled_modalities", [])
        variant_config["fusion"]["use_url_scalar_features"] = variant.get(
            "use_url_scalar_features", True
        )
        if "fusion_strategy" in variant:
            variant_config["fusion"]["strategy"] = variant["fusion_strategy"]

        model, model_type, model_name = build_fusion_model(variant_config)
        results = train_and_evaluate_model(
            model=model,
            model_type=model_type,
            model_name=f"{model_name} - {variant_name}",
            config=variant_config,
            device=device,
            df=df,
            features=features,
            output_dir=str(output_root / variant_name),
        )
        record = {
            "variant": variant_name,
            "fusion_strategy": variant_config["fusion"]["strategy"],
            "disabled_modalities": ",".join(variant_config["fusion"].get("disabled_modalities", [])),
            "use_url_scalar_features": variant_config["fusion"].get("use_url_scalar_features", True),
            "active_modalities": ",".join(results.get("active_modalities", [])),
            **flatten_metrics("val", results["val_metrics"]),
            **flatten_metrics("test", results["test_metrics"]),
            "checkpoint_path": results["checkpoint_path"],
            "optimal_threshold": results["optimal_threshold"],
            "calibration_temperature": results["calibration_temperature"],
        }
        records.append(record)
        if variant_name == "full_tuned":
            full_reference = record

    if full_reference is None and records:
        full_reference = records[0]

    for record in records:
        for metric_name in ["test_f1", "test_roc_auc", "test_c_index", "test_composite_score"]:
            record[f"delta_vs_full_{metric_name}"] = metric_delta(record, full_reference, metric_name)
            if baseline_metrics is not None:
                baseline_key = metric_name.replace("test_", "")
                record[f"delta_vs_baseline_{metric_name}"] = metric_delta(
                    record,
                    baseline_metrics,
                    baseline_key,
                )

    save_records(records, str(output_root), "ablation_results")
    df_records = pd.DataFrame(records)
    plot_ablation_metrics(df_records, output_root)
    plot_ablation_deltas(df_records, output_root)

    modality_contribution = (
        df_records[
            df_records["variant"].isin(["no_url", "no_text", "no_visual", "no_html"])
        ][["variant", "delta_vs_full_test_composite_score"]]
        .sort_values("delta_vs_full_test_composite_score")
        .to_dict(orient="records")
    )
    save_summary(
        {
            "baseline_metrics": baseline_metrics,
            "full_tuned": full_reference,
            "modality_contribution_ranking": modality_contribution,
        },
        str(output_root),
        "summary.json",
    )
    logger.info("Saved ablation artifacts to %s", output_root)


if __name__ == "__main__":
    main()
