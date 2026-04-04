#!/usr/bin/env python3
"""
Train unimodal baselines (URL-only, Text-only, Visual-only).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pathlib import Path
from transformers import AutoTokenizer

from src.utils.helpers import load_config, set_seed, get_device, setup_logging, count_parameters
from src.data.dataset_loader import PhishingDatasetLoader
from src.data.preprocessor import create_dataloaders, compute_class_weights
from src.models.fusion_model import URLOnlyClassifier, TextOnlyClassifier, VisualOnlyClassifier
from src.training.trainer import Trainer
from src.evaluation.metrics import collect_predictions, compute_metrics
from src.evaluation.analysis import plot_training_curves


def train_unimodal(config, model_cls, model_type, model_name, df, tokenizer, device):
    """Train a single unimodal model."""
    logger = setup_logging(config["project"]["output_dir"])
    output_dir = os.path.join(config["project"]["output_dir"], model_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*60}")

    train_loader, val_loader, test_loader = create_dataloaders(df, tokenizer, config)
    class_weights = compute_class_weights(df, device)

    model = model_cls(config)
    logger.info(f"Trainable parameters: {count_parameters(model):,}")

    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        model_type=model_type,
    )

    history = trainer.fit(output_dir)
    plot_training_curves(history, output_dir, model_name)

    # Test evaluation
    y_true, y_pred, y_prob = collect_predictions(
        model, test_loader, device, model_type
    )
    test_metrics = compute_metrics(y_true, y_pred, y_prob)
    logger.info(f"\n{model_name} Test Results:")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v}")

    return model, test_metrics, history


def main():
    config = load_config()
    set_seed(config["project"]["seed"])
    device = get_device(config)

    tokenizer = AutoTokenizer.from_pretrained(config["text"]["model_name"])

    loader = PhishingDatasetLoader(config)
    df = loader.build_merged_dataset()

    # Limit dataset size if configured
    max_samples = config["data"].get("max_samples")
    if max_samples:
        df = df.head(max_samples)

    results = {}

    # URL-only
    model, metrics, history = train_unimodal(
        config, URLOnlyClassifier, "url", "URL-Only", df, tokenizer, device
    )
    results["URL-Only"] = {"metrics": metrics, "history": history}

    # Text-only (only if HTML data exists)
    if df["html_content"].notna().sum() > 100:
        model, metrics, history = train_unimodal(
            config, TextOnlyClassifier, "text", "Text-Only", df, tokenizer, device
        )
        results["Text-Only"] = {"metrics": metrics, "history": history}

    # Visual-only (only if screenshot data exists)
    if df["image_path"].notna().sum() > 100:
        model, metrics, history = train_unimodal(
            config, VisualOnlyClassifier, "visual", "Visual-Only", df, tokenizer, device
        )
        results["Visual-Only"] = {"metrics": metrics, "history": history}

    print("\n\nUnimodal Results Summary:")
    print("-" * 60)
    for name, data in results.items():
        m = data["metrics"]
        print(
            f"{name:15s} | F1: {m['f1']:.4f} | "
            f"ROC-AUC: {m['roc_auc']:.4f} | Acc: {m['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
