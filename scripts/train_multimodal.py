#!/usr/bin/env python3
"""
Train the multi-modal fusion model.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pathlib import Path
from transformers import AutoTokenizer

from src.utils.helpers import (
    load_config, set_seed, get_device, setup_logging, count_parameters,
)
from src.data.dataset_loader import PhishingDatasetLoader
from src.data.preprocessor import create_dataloaders, compute_class_weights
from src.models.fusion_model import FusionClassifier
from src.training.trainer import Trainer
from src.evaluation.metrics import collect_predictions, compute_metrics
from src.evaluation.analysis import plot_training_curves


def main():
    config = load_config()
    set_seed(config["project"]["seed"])
    device = get_device(config)
    logger = setup_logging(config["project"]["output_dir"])

    output_dir = os.path.join(config["project"]["output_dir"], "multimodal")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config["text"]["model_name"])

    # Load data
    loader = PhishingDatasetLoader(config)
    df = loader.build_merged_dataset()

    max_samples = config["data"].get("max_samples")
    if max_samples:
        df = df.head(max_samples)

    train_loader, val_loader, test_loader = create_dataloaders(
        df, tokenizer, config
    )
    class_weights = compute_class_weights(df, device)

    # Build model
    model = FusionClassifier(config)
    logger.info(f"Fusion strategy: {config['fusion']['strategy']}")
    logger.info(f"Trainable parameters: {count_parameters(model):,}")

    # Train
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        model_type="multimodal",
    )

    history = trainer.fit(output_dir)
    plot_training_curves(history, output_dir, "Multimodal Fusion")

    # Test evaluation
    y_true, y_pred, y_prob = collect_predictions(
        model, test_loader, device, "multimodal"
    )
    test_metrics = compute_metrics(y_true, y_pred, y_prob)

    logger.info("\nMultimodal Test Results:")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v}")

    # Log modality weights if using weighted fusion
    weights = model.get_modality_weights()
    if weights is not None:
        logger.info(
            f"\nLearned modality weights: "
            f"URL={weights[0]:.3f}, Text={weights[1]:.3f}, "
            f"Visual={weights[2]:.3f}"
        )


if __name__ == "__main__":
    main()
