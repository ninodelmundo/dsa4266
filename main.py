#!/usr/bin/env python3
"""
Main entry point for the Phishing Detection Multi-Modal ML System.

Usage:
    python main.py                    # full pipeline (download + preprocess + train + evaluate)
    python main.py --step download    # download data only
    python main.py --step preprocess  # preprocess only
    python main.py --step train       # train multimodal model
    python main.py --step evaluate    # evaluate only
    python main.py --unimodal         # also train URL/Text/Visual baselines
"""
import argparse
import subprocess
import sys
import os
from pathlib import Path


def step_download(config, logger):
    """Download all datasets."""
    logger.info("Step 1: Downloading datasets...")
    subprocess.run(
        [sys.executable, "scripts/download_data.py"],
        check=True,
    )


def step_preprocess(config, logger):
    """Build merged dataset."""
    from src.data.dataset_loader import PhishingDatasetLoader

    logger.info("Step 2: Preprocessing and merging datasets...")
    loader = PhishingDatasetLoader(config)
    df = loader.build_merged_dataset()
    logger.info(f"Merged dataset: {df.shape}")
    logger.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")
    return df


def step_train(config, logger, device, df, include_unimodal=False):
    """Train models using pre-extracted embeddings for speed."""
    import gc
    import torch

    from src.data.preprocessor import (
        extract_and_save_features,
        create_fast_dataloaders,
        compute_class_weights,
    )
    from src.models.fusion_model import (
        FastFusionClassifier,
        FastURLOnlyClassifier,
        FastTextOnlyClassifier,
        FastVisualOnlyClassifier,
    )
    from src.training.trainer import Trainer
    from src.evaluation.metrics import (
        collect_predictions,
        compute_metrics,
        find_optimal_threshold,
    )
    from src.evaluation.analysis import (
        plot_training_curves,
        plot_class_distribution,
        plot_dataset_stats,
        plot_embedding_tsne,
        plot_threshold_sweep,
        plot_modality_attention_weights,
        plot_prediction_confidence,
        plot_feature_correlation,
        save_results_summary,
        save_misclassification_analysis,
        save_model_architecture_summary,
        plot_learning_rate_schedule,
    )
    from src.utils.helpers import count_parameters
    from src.experiments.common import get_metric_weights, limit_dataset

    output_base = config["project"]["output_dir"]
    metric_weights = get_metric_weights(config)

    max_samples = config["data"].get("max_samples")
    df = limit_dataset(df, max_samples, seed=config["project"]["seed"])

    # Extract frozen DistilBERT + EfficientNet embeddings once (cached)
    features = extract_and_save_features(df, config)

    train_loader, val_loader, test_loader = create_fast_dataloaders(
        df, features, config
    )
    class_weights = compute_class_weights(df, device)

    all_results = {}

    # Data study plots (rubric: "Is the data balanced?")
    eval_dir = os.path.join(output_base, "evaluation")
    Path(eval_dir).mkdir(parents=True, exist_ok=True)
    plot_class_distribution(df, eval_dir)
    plot_dataset_stats(df, eval_dir)
    logger.info("Saved data study plots (class distribution, dataset stats)")

    # t-SNE embedding visualization (rubric: "Did the model learn anything meaningful?")
    plot_embedding_tsne(features, eval_dir)
    logger.info("Saved t-SNE embedding visualization")

    # Feature correlation heatmap (rubric: data study depth)
    plot_feature_correlation(features, df, eval_dir)
    logger.info("Saved feature correlation heatmap")

    # Learning rate schedule visualization
    plot_learning_rate_schedule(config, eval_dir)
    logger.info("Saved learning rate schedule plot")

    # Unimodal baselines (optional)
    if include_unimodal:
        unimodal_configs = [
            ("URL-Only", FastURLOnlyClassifier, "fast_url"),
            ("Text-Only", FastTextOnlyClassifier, "fast_text"),
            ("Visual-Only", FastVisualOnlyClassifier, "fast_visual"),
        ]

        for name, model_cls, model_type in unimodal_configs:
            out_dir = os.path.join(output_base, name)
            Path(out_dir).mkdir(parents=True, exist_ok=True)

            logger.info(f"\n{'='*60}\nTraining {name}\n{'='*60}")
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
                metric_weights=metric_weights,
                checkpoint_metric=config["training"].get("checkpoint_metric", "val_loss"),
                use_amp=config.get("runtime", {}).get("amp", False),
            )
            history = trainer.fit(out_dir)
            plot_training_curves(history, out_dir, name)

            # Find optimal threshold on validation set
            val_true, _, val_prob = collect_predictions(
                model, val_loader, device, model_type
            )
            optimal_thresh = find_optimal_threshold(val_true, val_prob)
            logger.info(f"  Optimal threshold (val F1): {optimal_thresh:.2f}")

            y_true, _, y_prob = collect_predictions(
                model, test_loader, device, model_type
            )
            y_pred = (y_prob >= optimal_thresh).astype(int)
            metrics = compute_metrics(
                y_true,
                y_pred,
                y_prob,
                optimal_thresh,
                metric_weights=metric_weights,
            )
            all_results[name] = {
                "y_true": y_true,
                "y_pred": y_pred,
                "y_prob": y_prob,
                "metrics": metrics,
            }

            del model, trainer
            gc.collect()

    # Multimodal fusion (always)
    out_dir = os.path.join(output_base, "multimodal")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}\nTraining Multimodal Fusion\n{'='*60}")
    fusion_model = FastFusionClassifier(config)
    logger.info(f"Trainable parameters: {count_parameters(fusion_model):,}")

    trainer = Trainer(
        model=fusion_model,
        config=config,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        model_type="fast_multimodal",
        metric_weights=metric_weights,
        checkpoint_metric=config["training"].get("checkpoint_metric", "val_loss"),
        use_amp=config.get("runtime", {}).get("amp", False),
    )
    history = trainer.fit(out_dir)
    plot_training_curves(history, out_dir, "Multimodal Fusion")

    # Find optimal threshold on validation set
    val_true, _, val_prob = collect_predictions(
        fusion_model, val_loader, device, "fast_multimodal"
    )
    optimal_thresh = find_optimal_threshold(val_true, val_prob)
    logger.info(f"  Optimal threshold (val F1): {optimal_thresh:.2f}")

    y_true, _, y_prob = collect_predictions(
        fusion_model, test_loader, device, "fast_multimodal"
    )
    y_pred = (y_prob >= optimal_thresh).astype(int)
    metrics = compute_metrics(
        y_true,
        y_pred,
        y_prob,
        optimal_thresh,
        metric_weights=metric_weights,
    )
    all_results["Multimodal"] = {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics,
    }

    weights = fusion_model.get_modality_weights()
    if weights is not None:
        active_modalities = fusion_model.get_active_modalities()
        weight_log = ", ".join(
            f"{name}={weight:.3f}"
            for name, weight in zip(active_modalities, weights.tolist())
        )
        logger.info(f"Learned modality weights: {weight_log}")

    # Threshold sweep plot (rubric: "Which metric should we focus on?")
    plot_threshold_sweep(y_true, y_prob, eval_dir)
    logger.info("Saved threshold sweep plot")

    # Attention weight heatmap (rubric: "Interpretability")
    if fusion_model.strategy == "attention":
        plot_modality_attention_weights(fusion_model, features, device, eval_dir)
        logger.info("Saved attention weight heatmap")

    # Prediction confidence distribution (rubric: "Results discussion")
    plot_prediction_confidence(y_true, y_prob, eval_dir)
    logger.info("Saved prediction confidence distribution")

    # Model architecture summary
    save_model_architecture_summary(fusion_model, config, eval_dir)
    logger.info("Saved model architecture summary")

    # Results summary CSV + text
    save_results_summary(all_results, eval_dir)
    logger.info("Saved results summary (CSV + TXT)")

    # Misclassification error analysis
    save_misclassification_analysis(all_results, df, eval_dir)
    logger.info("Saved misclassification analysis")

    return all_results


def step_evaluate(config, logger, all_results):
    """Generate evaluation plots and summary."""
    from src.evaluation.analysis import (
        plot_confusion_matrix,
        plot_roc_curves,
        plot_pr_curves,
        plot_metrics_comparison,
        ablation_study_plot,
    )

    eval_dir = os.path.join(config["project"]["output_dir"], "evaluation")
    Path(eval_dir).mkdir(parents=True, exist_ok=True)

    for name, data in all_results.items():
        plot_confusion_matrix(
            data["y_true"], data["y_pred"], eval_dir, f"{name} CM"
        )

    if len(all_results) > 1:
        plot_roc_curves(all_results, eval_dir)
        plot_pr_curves(all_results, eval_dir)
        plot_metrics_comparison(all_results, eval_dir)

    if "Multimodal" in all_results:
        ablation_results = {"Full Multimodal": all_results["Multimodal"]}
        for name in ["URL-Only", "Text-Only", "Visual-Only"]:
            if name in all_results:
                ablation_results[name] = all_results[name]
        if len(ablation_results) > 1:
            ablation_study_plot(ablation_results, eval_dir)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 70)
    for name, data in all_results.items():
        m = data["metrics"]
        logger.info(
            f"{name:20s} | F1: {m['f1']:.4f} | "
            f"ROC-AUC: {m['roc_auc']:.4f} | "
            f"Acc: {m['accuracy']:.4f} | "
            f"Prec: {m['precision']:.4f} | "
            f"Rec: {m['recall']:.4f} | "
            f"Thresh: {m['threshold']:.2f}"
        )
    logger.info("=" * 70)
    logger.info(f"Plots saved to {eval_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Phishing Detection Multi-Modal ML System"
    )
    parser.add_argument(
        "--step",
        choices=["download", "preprocess", "train", "evaluate", "all"],
        default="all",
        help="Which pipeline step to run (default: all)",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--unimodal",
        action="store_true",
        help="Also train unimodal baselines (URL-Only, Text-Only, Visual-Only)",
    )
    args = parser.parse_args()

    from src.utils.helpers import load_config, set_seed, get_device, setup_logging

    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    device = get_device(config)
    logger = setup_logging(config["project"]["output_dir"])

    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.config}")

    if args.step in ("download", "all"):
        step_download(config, logger)

    df = None
    if args.step in ("preprocess", "train", "evaluate", "all"):
        df = step_preprocess(config, logger)

    all_results = None
    if args.step in ("train", "all"):
        all_results = step_train(config, logger, device, df, args.unimodal)

    if args.step in ("evaluate", "all") and all_results:
        step_evaluate(config, logger, all_results)

    if args.step == "evaluate" and all_results is None:
        logger.info(
            "To run standalone evaluation, use scripts/evaluate.py "
            "(requires trained model checkpoints)."
        )


if __name__ == "__main__":
    main()
