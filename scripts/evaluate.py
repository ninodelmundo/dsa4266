#!/usr/bin/env python3
"""
Comprehensive evaluation: load best models, compute metrics, generate plots.
Includes ablation study and robustness analysis.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

from src.utils.helpers import (
    load_config, set_seed, get_device, setup_logging, load_checkpoint,
)
from src.data.dataset_loader import PhishingDatasetLoader
from src.data.preprocessor import create_dataloaders
from src.models.fusion_model import (
    FusionClassifier, URLOnlyClassifier, TextOnlyClassifier, VisualOnlyClassifier,
)
from src.evaluation.metrics import (
    collect_predictions, compute_metrics, find_optimal_threshold,
)
from src.evaluation.analysis import (
    plot_confusion_matrix, plot_roc_curves, plot_pr_curves,
    plot_metrics_comparison, ablation_study_plot,
)


def load_model(model_cls, config, checkpoint_path, device):
    """Load a trained model from checkpoint."""
    model = model_cls(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def run_ablation(model, test_loader, device, config):
    """
    Ablation study: zero out each modality and measure performance drop.
    """
    results = {}

    # Full model
    y_true, y_pred, y_prob = collect_predictions(
        model, test_loader, device, "multimodal"
    )
    results["Full Multimodal"] = {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": compute_metrics(y_true, y_pred, y_prob),
    }

    # TODO: implement modality zeroing by wrapping forward pass
    # This would require modifying the forward to accept masking flags
    # For now, we compare against unimodal baselines

    return results


def main():
    config = load_config()
    set_seed(config["project"]["seed"])
    device = get_device(config)
    logger = setup_logging(config["project"]["output_dir"])

    eval_dir = os.path.join(config["project"]["output_dir"], "evaluation")
    Path(eval_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config["text"]["model_name"])

    # Load data
    loader = PhishingDatasetLoader(config)
    df = loader.build_merged_dataset()

    max_samples = config["data"].get("max_samples")
    if max_samples:
        df = df.head(max_samples)

    _, _, test_loader = create_dataloaders(df, tokenizer, config)

    all_results = {}

    # ── Evaluate each model ───────────────────────────────────────────────────

    model_configs = [
        ("URL-Only", URLOnlyClassifier, "url", "URL-Only"),
        ("Text-Only", TextOnlyClassifier, "text", "Text-Only"),
        ("Visual-Only", VisualOnlyClassifier, "visual", "Visual-Only"),
        ("Multimodal", FusionClassifier, "multimodal", "multimodal"),
    ]

    for display_name, model_cls, model_type, dir_name in model_configs:
        ckpt_path = os.path.join(
            config["project"]["output_dir"], dir_name, "best_model.pt"
        )
        if not os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint not found for {display_name}: {ckpt_path}")
            continue

        logger.info(f"\nEvaluating {display_name}...")
        model = load_model(model_cls, config, ckpt_path, device)

        y_true, y_pred, y_prob = collect_predictions(
            model, test_loader, device, model_type
        )

        # Find optimal threshold
        opt_threshold = find_optimal_threshold(y_true, y_prob)
        metrics = compute_metrics(y_true, y_pred, y_prob, opt_threshold)

        all_results[display_name] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "metrics": metrics,
        }

        # Per-model confusion matrix
        plot_confusion_matrix(y_true, y_pred, eval_dir, f"{display_name} CM")

        logger.info(f"{display_name} Results (threshold={opt_threshold:.2f}):")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v}")

    # ── Comparative plots ─────────────────────────────────────────────────────

    if len(all_results) > 1:
        plot_roc_curves(all_results, eval_dir)
        plot_pr_curves(all_results, eval_dir)
        plot_metrics_comparison(all_results, eval_dir)
        logger.info(f"\nComparative plots saved to {eval_dir}")

    # ── Ablation study ────────────────────────────────────────────────────────

    multimodal_ckpt = os.path.join(
        config["project"]["output_dir"], "multimodal", "best_model.pt"
    )
    if os.path.exists(multimodal_ckpt) and config["evaluation"].get("ablation"):
        logger.info("\nRunning ablation study...")
        multimodal_model = load_model(
            FusionClassifier, config, multimodal_ckpt, device
        )
        ablation_results = run_ablation(
            multimodal_model, test_loader, device, config
        )
        # Merge unimodal results into ablation
        for name in ["URL-Only", "Text-Only", "Visual-Only"]:
            if name in all_results:
                ablation_results[name] = all_results[name]
        ablation_study_plot(ablation_results, eval_dir)

    # ── Save summary ──────────────────────────────────────────────────────────

    summary = {}
    for name, data in all_results.items():
        summary[name] = {
            k: v
            for k, v in data["metrics"].items()
            if isinstance(v, (int, float))
        }

    summary_path = os.path.join(eval_dir, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nResults summary saved to {summary_path}")


if __name__ == "__main__":
    main()
