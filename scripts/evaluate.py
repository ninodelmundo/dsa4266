#!/usr/bin/env python3
"""
Comprehensive evaluation: load the best baseline and Optuna-tuned
checkpoints, compute metrics on the held-out test split, write per-model
confusion matrices plus comparative ROC/PR/metrics plots, and emit a
unified comparison summary.

Reads from the standardised directory trees produced by the training
scripts:

    outputs/baseline/unimodal/{url,text,visual,html}/summary.json
    outputs/baseline/fusion/summary.json
    outputs/optimization/unimodal/{url,text,visual,html}/summary.json
    outputs/optimization/fusion/summary.json

and writes:

    outputs/evaluation/
        *_cm.png                          (one per evaluated model)
        roc_curves.png  pr_curves.png     (comparative)
        metrics_comparison.png
        results_summary.{csv,txt}         (legacy human-readable)
        summary.json                      (machine-readable comparison)
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")

import torch

from src.experiments.common import (
    apply_overrides,
    build_fast_loaders_and_weights,
    clone_config,
    flatten_metrics,
    load_fast_resources,
    prepare_environment,
    save_summary,
)
from src.experiments.tuning import build_fusion_model, build_unimodal_model
from src.evaluation.metrics import (
    calibrate_temperature,
    collect_predictions_calibrated,
    compute_metrics,
    find_optimal_threshold,
)
from src.evaluation.analysis import (
    ablation_study_plot,
    plot_confusion_matrix,
    plot_fusion_tsne_comparison,
    plot_metrics_comparison,
    plot_pr_curves,
    plot_roc_curves,
)

MODALITIES = ["url", "text", "visual", "html"]
DISPLAY_NAME = {
    "url": "URL-Only",
    "text": "Text-Only",
    "visual": "Visual-Only",
    "html": "HTML-Only",
    "fusion": "Multimodal",
}


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _load_unimodal_run(config: dict, source: str, modality: str, device):
    """Load a unimodal run from outputs/{source}/unimodal/{modality}/summary.json."""
    output_dir = Path(config["project"]["output_dir"]) / source / "unimodal" / modality
    summary = _read_json(output_dir / "summary.json")
    if summary is None:
        return None
    best_record = summary.get("best_record", {})
    checkpoint_path = best_record.get("checkpoint_path")
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return None

    run_config = clone_config(config)
    apply_overrides(run_config, _modality_overrides(summary.get("best_params", {}), modality))
    model, model_type, model_name = build_unimodal_model(modality, run_config)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return {
        "run_config": run_config,
        "model": model,
        "model_type": model_type,
        "model_name": model_name,
        "summary": summary,
        "checkpoint_path": checkpoint_path,
    }


def _load_fusion_run(config: dict, source: str, device):
    """Load a fusion run from outputs/{source}/fusion/summary.json.

    For the Optuna-tuned source, the fusion checkpoint depends on both the
    tuned fusion hyperparameters AND the URL-encoder architecture that was
    promoted from the unimodal study. Those combined overrides are stored in
    outputs/{source}/fusion/best_fusion_overrides.json; prefer that file
    when present so the rebuilt model matches the checkpoint's state_dict.
    """
    output_dir = Path(config["project"]["output_dir"]) / source / "fusion"
    summary = _read_json(output_dir / "summary.json")
    if summary is None:
        return None
    best_record = summary.get("best_record", {})
    checkpoint_path = best_record.get("checkpoint_path")
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return None

    run_config = clone_config(config)
    combined_overrides = _read_json(output_dir / "best_fusion_overrides.json")
    if combined_overrides:
        apply_overrides(run_config, combined_overrides)
    else:
        apply_overrides(run_config, _fusion_overrides(summary.get("best_params", {})))
    model, model_type, model_name = build_fusion_model(run_config)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return {
        "run_config": run_config,
        "model": model,
        "model_type": model_type,
        "model_name": model_name,
        "summary": summary,
        "checkpoint_path": checkpoint_path,
    }


def _modality_overrides(best_params: dict, modality: str) -> dict:
    overrides = {"training": {}, modality: {}, "fusion": {}}
    for key in ["batch_size", "optimizer", "scheduler", "sampling_strategy", "class_weights", "learning_rate", "weight_decay"]:
        if key in best_params:
            overrides["training"][key] = best_params[key]
    if "fusion_dropout" in best_params:
        overrides["fusion"]["dropout"] = best_params["fusion_dropout"]
    for key in ["model_type", "embedding_dim", "hidden_dim", "dropout",
                "classifier_hidden_dim", "classifier_bottleneck_dim",
                "use_url_scalar_features"]:
        if key in best_params:
            overrides[modality][key] = best_params[key]
    return {section: values for section, values in overrides.items() if values}


def _fusion_overrides(best_params: dict) -> dict:
    overrides = {"training": {}, "fusion": {}}
    for key in ["batch_size", "optimizer", "scheduler", "sampling_strategy", "class_weights", "learning_rate", "weight_decay"]:
        if key in best_params:
            overrides["training"][key] = best_params[key]
    for key in ["strategy", "projected_dim", "hidden_dim", "dropout"]:
        if key in best_params:
            overrides["fusion"][key] = best_params[key]
    return {section: values for section, values in overrides.items() if values}


def _evaluate_run(run: dict, test_loader, val_loader, device) -> Dict:
    """Recompute test metrics from a loaded checkpoint on the same test loader."""
    model = run["model"]
    model_type = run["model_type"]
    temperature = calibrate_temperature(model, val_loader, device, model_type)
    val_true, _, val_prob = collect_predictions_calibrated(
        model, val_loader, device, model_type, temperature=temperature,
    )
    threshold = float(find_optimal_threshold(val_true, val_prob))

    test_true, _, test_prob = collect_predictions_calibrated(
        model, test_loader, device, model_type, temperature=temperature,
    )
    test_pred = (test_prob >= threshold).astype(int)
    test_metrics = compute_metrics(test_true, test_pred, test_prob, threshold)

    return {
        "model_name": run["model_name"],
        "model_type": model_type,
        "checkpoint_path": run["checkpoint_path"],
        "optimal_threshold": threshold,
        "calibration_temperature": temperature,
        "y_true": test_true,
        "y_pred": test_pred,
        "y_prob": test_prob,
        "metrics": test_metrics,
    }


def _write_human_summary(eval_dir: Path, rows):
    lines = ["=" * 80, "PHISHING DETECTION - FINAL RESULTS SUMMARY", "=" * 80, ""]
    for row in rows:
        m = row["metrics"]
        lines.append(f"Model: {row['model_name']}  [{row['source']}]")
        lines.append(
            f"  F1: {m['f1']:.4f}  |  ROC-AUC: {m['roc_auc']:.4f}  |  Accuracy: {m['accuracy']:.4f}"
        )
        lines.append(
            f"  Precision: {m['precision']:.4f}  |  Recall: {m['recall']:.4f}  |  PR-AUC: {m['pr_auc']:.4f}"
        )
        lines.append(f"  Threshold: {row['optimal_threshold']:.2f}")
        lines.append(
            f"  Confusion: TP={m.get('tp','?')}  FP={m.get('fp','?')}  FN={m.get('fn','?')}  TN={m.get('tn','?')}"
        )
        lines.append("")
    (eval_dir / "results_summary.txt").write_text("\n".join(lines))

    import pandas as pd
    csv_rows = []
    for row in rows:
        m = row["metrics"]
        csv_rows.append({
            "model": row["model_name"],
            "source": row["source"],
            "threshold": row["optimal_threshold"],
            **{k: v for k, v in m.items() if isinstance(v, (int, float))},
        })
    pd.DataFrame(csv_rows).to_csv(eval_dir / "results_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline and Optuna-tuned models on the held-out test split.",
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file.")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["baseline", "optimization"],
        help="Which run trees to evaluate (relative to outputs/).",
    )
    args = parser.parse_args()

    config, device, logger = prepare_environment(args.config)
    df, features = load_fast_resources(config)
    _, val_loader, test_loader, _ = build_fast_loaders_and_weights(config, df, features, device)

    eval_dir = Path(config["project"]["output_dir"]) / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    fusion_tsne_run = None
    for source in args.sources:
        logger.info("Evaluating runs from source=%s", source)
        for modality in MODALITIES:
            run = _load_unimodal_run(config, source, modality, device)
            if run is None:
                logger.warning("No %s/%s run found; skipping.", source, modality)
                continue
            logger.info("Evaluating %s (%s) ...", DISPLAY_NAME[modality], source)
            evaluated = _evaluate_run(run, test_loader, val_loader, device)
            evaluated["source"] = source
            evaluated["modality"] = modality
            plot_confusion_matrix(
                evaluated["y_true"], evaluated["y_pred"], str(eval_dir),
                f"{DISPLAY_NAME[modality]} ({source}) CM",
            )
            rows.append(evaluated)

        fusion_run = _load_fusion_run(config, source, device)
        if fusion_run is None:
            logger.warning("No %s fusion run found; skipping.", source)
            continue
        logger.info("Evaluating Multimodal (%s) ...", source)
        evaluated = _evaluate_run(fusion_run, test_loader, val_loader, device)
        evaluated["source"] = source
        evaluated["modality"] = "fusion"
        plot_confusion_matrix(
            evaluated["y_true"], evaluated["y_pred"], str(eval_dir),
            f"Multimodal ({source}) CM",
        )
        rows.append(evaluated)
        if fusion_tsne_run is None or source == "optimization":
            fusion_tsne_run = {
                "source": source,
                "model": fusion_run["model"],
            }

    if not rows:
        logger.error("No runs found to evaluate. Run the training or optimization scripts first.")
        return

    # Comparative plots
    all_results = {
        f"{row['model_name']} [{row['source']}]": {
            "y_true": row["y_true"],
            "y_pred": row["y_pred"],
            "y_prob": row["y_prob"],
            "metrics": row["metrics"],
        }
        for row in rows
    }
    if len(all_results) > 1:
        plot_roc_curves(all_results, str(eval_dir))
        plot_pr_curves(all_results, str(eval_dir))
        plot_metrics_comparison(all_results, str(eval_dir))

    # Ablation-style comparison (baseline vs tuned) if both are present
    sources_present = {row["source"] for row in rows}
    if {"baseline", "optimization"}.issubset(sources_present):
        ablation_study_plot(all_results, str(eval_dir))

    if fusion_tsne_run is not None:
        logger.info(
            "Generating learned-fusion t-SNE comparison from source=%s",
            fusion_tsne_run["source"],
        )
        try:
            plot_fusion_tsne_comparison(
                fusion_tsne_run["model"],
                test_loader,
                str(eval_dir),
                source_name=fusion_tsne_run["source"],
            )
        except Exception as exc:
            logger.warning("Could not generate learned-fusion t-SNE comparison: %s", exc)

    # Machine-readable summary
    summary_rows = []
    for row in rows:
        summary_rows.append({
            "source": row["source"],
            "modality": row["modality"],
            "model_name": row["model_name"],
            "checkpoint_path": row["checkpoint_path"],
            "optimal_threshold": row["optimal_threshold"],
            "calibration_temperature": row["calibration_temperature"],
            **flatten_metrics("test", {k: v for k, v in row["metrics"].items() if isinstance(v, (int, float))}),
            "confusion_matrix": {
                "tp": row["metrics"].get("tp"),
                "fp": row["metrics"].get("fp"),
                "fn": row["metrics"].get("fn"),
                "tn": row["metrics"].get("tn"),
            },
        })
    save_summary({"results": summary_rows}, str(eval_dir), "summary.json")
    _write_human_summary(eval_dir, rows)
    logger.info("Saved evaluation artifacts to %s", eval_dir)


if __name__ == "__main__":
    main()
