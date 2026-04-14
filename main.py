#!/usr/bin/env python3
"""
Main entry point for the Phishing Detection Multi-Modal ML System.

This is a thin orchestrator that runs the individual scripts in the
correct order.  Each script writes to a standardised output tree so the
baseline and Optuna-tuned results can be compared apples-to-apples:

    outputs/baseline/unimodal/{url,text,visual,html}/summary.json
    outputs/baseline/fusion/summary.json
    outputs/optimization/unimodal/{url,text,visual,html}/summary.json
    outputs/optimization/fusion/summary.json
    outputs/optimization/ablation/ablation_results.{csv,json}
    outputs/evaluation/{summary.json, results_summary.{txt,csv}, *.png}
    outputs/explainability/...

Usage
-----
    # Full default pipeline: download -> preprocess -> aux plots ->
    # baseline unimodal -> baseline fusion -> evaluate.
    python main.py

    # Include Optuna tuning (slow; many trials) and ablation.
    python main.py --optimize --ablation

    # Include explainability (requires a trained fusion checkpoint).
    python main.py --explain

    # Just one step.
    python main.py --step download
    python main.py --step aux          # regenerate EDA / data-study plots
    python main.py --step baseline     # baseline unimodal + fusion
    python main.py --step optimize     # optimize_unimodal + optimize_fusion
    python main.py --step ablation     # run_ablation
    python main.py --step evaluate     # scripts/evaluate.py over all sources
    python main.py --step explain      # scripts/run_explainability.py
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"


def _run(cmd, logger, description):
    logger.info("== %s ==", description)
    logger.info("$ %s", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def step_download(logger):
    _run([sys.executable, str(SCRIPTS / "download_data.py")], logger, "Download datasets")


def step_preprocess(config, logger):
    from src.data.dataset_loader import PhishingDatasetLoader

    logger.info("== Preprocess / merge datasets ==")
    loader = PhishingDatasetLoader(config)
    df = loader.build_merged_dataset()
    logger.info("Merged dataset: %s", df.shape)
    logger.info("Label distribution:\n%s", df["label"].value_counts().to_string())
    return df


def step_aux_plots(config, logger, df):
    """Generate the report figures that don't depend on a trained model
    (class distribution, dataset stats, feature correlation, t-SNE,
    learning-rate schedule).  These populate Figs. 1-3 of the report."""
    from src.data.preprocessor import extract_and_save_features
    from src.evaluation.analysis import (
        plot_class_distribution,
        plot_dataset_stats,
        plot_embedding_tsne,
        plot_feature_correlation,
        plot_learning_rate_schedule,
    )
    from src.experiments.common import limit_dataset

    logger.info("== Auxiliary data-study plots ==")
    df = limit_dataset(df, config["data"].get("max_samples"), seed=config["project"]["seed"])
    features = extract_and_save_features(df, config)

    eval_dir = os.path.join(config["project"]["output_dir"], "evaluation")
    Path(eval_dir).mkdir(parents=True, exist_ok=True)

    plot_class_distribution(df, eval_dir)
    plot_dataset_stats(df, eval_dir)
    plot_embedding_tsne(features, eval_dir)
    plot_feature_correlation(features, df, eval_dir)
    plot_learning_rate_schedule(config, eval_dir)
    logger.info("Saved data-study plots to %s", eval_dir)


def step_baseline(logger):
    _run(
        [sys.executable, str(SCRIPTS / "train_unimodal.py"), "--modality", "all"],
        logger, "Baseline unimodal training (all 4 modalities)",
    )
    _run(
        [sys.executable, str(SCRIPTS / "train_multimodal.py")],
        logger, "Baseline fusion training",
    )


def step_optimize(logger):
    _run(
        [sys.executable, str(SCRIPTS / "optimize_unimodal.py"), "--modality", "all"],
        logger, "Optuna unimodal search (all 4 modalities)",
    )
    _run(
        [sys.executable, str(SCRIPTS / "optimize_fusion.py")],
        logger, "Optuna fusion search (uses promoted unimodal overrides)",
    )


def step_ablation(logger):
    _run(
        [sys.executable, str(SCRIPTS / "run_ablation.py")],
        logger, "Ablation over tuned fusion",
    )


def step_evaluate(logger, sources):
    cmd = [sys.executable, str(SCRIPTS / "evaluate.py"), "--sources", *sources]
    _run(cmd, logger, "Unified evaluation across sources=%s" % ",".join(sources))


def step_explain(logger):
    _run(
        [sys.executable, str(SCRIPTS / "run_explainability.py")],
        logger, "Explainability (SHAP + modality Shapley)",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Phishing Detection Multi-Modal ML System - full pipeline orchestrator.",
    )
    parser.add_argument(
        "--step",
        choices=["download", "preprocess", "aux", "baseline", "optimize", "ablation", "evaluate", "explain", "all"],
        default="all",
        help="Which pipeline step to run (default: all)",
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--optimize", action="store_true", help="When --step all, also run Optuna tuning.")
    parser.add_argument("--ablation", action="store_true", help="When --step all, also run the ablation study.")
    parser.add_argument("--explain", action="store_true", help="When --step all, also run explainability.")
    parser.add_argument("--skip-download", action="store_true", help="Skip the download step when --step all.")
    args = parser.parse_args()

    from src.utils.helpers import load_config, set_seed, get_device, setup_logging

    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    device = get_device(config)
    logger = setup_logging(config["project"]["output_dir"])

    logger.info("Device: %s", device)
    logger.info("Config: %s", args.config)

    run_all = args.step == "all"

    if args.step == "download" or (run_all and not args.skip_download):
        step_download(logger)

    df = None
    if args.step in {"preprocess", "aux", "all"}:
        df = step_preprocess(config, logger)

    if args.step == "aux" or run_all:
        step_aux_plots(config, logger, df)

    if args.step == "baseline" or run_all:
        step_baseline(logger)

    if args.step == "optimize" or (run_all and args.optimize):
        step_optimize(logger)

    if args.step == "ablation" or (run_all and args.ablation):
        step_ablation(logger)

    if args.step == "evaluate" or run_all:
        sources = ["baseline"]
        if Path("outputs/optimization").exists():
            if any(Path("outputs/optimization").glob("*/summary.json")) or any(
                Path("outputs/optimization").glob("*/*/summary.json")
            ):
                sources.append("optimization")
        step_evaluate(logger, sources)

    if args.step == "explain" or (run_all and args.explain):
        step_explain(logger)


if __name__ == "__main__":
    main()
