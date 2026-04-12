#!/usr/bin/env python3
"""Run explainability analysis for the fast cached-feature pipeline."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")

from src.experiments.common import load_fast_resources, prepare_environment
from src.explainability.engine import (
    build_dashboard_frame,
    compute_modality_shapley,
    compute_surrogate_shap,
    export_shapash_explainer,
    fit_score_surrogate,
    load_fusion_model,
    load_unimodal_models,
    prepare_output_dir,
    predict_loaded_model,
    save_dashboard_artifacts,
    save_explainability_report,
    save_local_explanations,
    save_modality_contributions,
    save_summary,
    select_split_samples,
)


def main():
    parser = argparse.ArgumentParser(
        description="Explain fast unimodal and multimodal phishing models."
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file.")
    parser.add_argument("--split", default=None, help="Dataset split to explain: train, val, test, or all.")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to explain.")
    parser.add_argument("--background-samples", type=int, default=None, help="Background samples for Shapley replacement values.")
    parser.add_argument("--local-samples", type=int, default=None, help="Number of local sample reports to create.")
    args = parser.parse_args()

    config, device, logger = prepare_environment(args.config)
    xai_cfg = config.get("explainability", {})
    split = args.split or xai_cfg.get("split", "test")
    max_samples = args.max_samples if args.max_samples is not None else xai_cfg.get("max_samples", 100)
    background_samples = (
        args.background_samples
        if args.background_samples is not None
        else xai_cfg.get("background_samples", 50)
    )
    local_samples = args.local_samples if args.local_samples is not None else xai_cfg.get("local_samples", 20)
    seed = int(xai_cfg.get("seed", config["project"].get("seed", 42)))
    methods = xai_cfg.get("methods", {})

    output_dir = prepare_output_dir(Path(xai_cfg.get("output_dir", "outputs/explainability")))
    logger.info("Saving explainability artifacts to %s", output_dir)

    df, features = load_fast_resources(config)
    sample_df, selected_features = select_split_samples(
        df,
        features,
        split=split,
        max_samples=max_samples,
        seed=seed,
    )
    logger.info("Selected %d samples from split=%s", len(sample_df), split)

    fusion = load_fusion_model(config, device, logger)
    unimodal = load_unimodal_models(config, device, logger)

    fusion_probs = predict_loaded_model(fusion, selected_features, device)
    unimodal_probs = {
        modality: predict_loaded_model(loaded, selected_features, device)
        for modality, loaded in unimodal.items()
    }

    dashboard_frame = build_dashboard_frame(
        sample_df,
        selected_features,
        fusion_probs,
        unimodal_probs,
        threshold=fusion.threshold,
    )
    surrogate, surrogate_x, surrogate_pred, fidelity = fit_score_surrogate(
        dashboard_frame,
        seed=seed,
    )
    save_dashboard_artifacts(dashboard_frame, surrogate_x, surrogate_pred, output_dir)
    importance, shap_values = compute_surrogate_shap(
        surrogate,
        surrogate_x,
        output_dir,
        logger,
    )
    shapash_path = None
    if xai_cfg.get("shapash", {}).get("enabled", True):
        shapash_path = export_shapash_explainer(
            surrogate,
            surrogate_x,
            surrogate_pred,
            output_dir,
            logger,
        )

    modality_local, modality_global = compute_modality_shapley(
        fusion,
        selected_features,
        device,
        background_samples=background_samples,
        local_samples=local_samples,
    )
    save_modality_contributions(modality_local, modality_global, output_dir)

    save_local_explanations(
        sample_df,
        selected_features,
        dashboard_frame,
        surrogate_x,
        shap_values,
        modality_local,
        unimodal,
        output_dir,
        device,
        logger,
        local_samples=local_samples,
        methods=methods,
    )
    report_path = save_explainability_report(
        output_dir,
        dashboard_frame,
        importance,
        modality_global,
        fidelity,
        shapash_path,
        local_samples=local_samples,
    )
    save_summary(
        output_dir,
        fusion,
        unimodal,
        fidelity,
        shapash_path,
        report_path,
        sample_count=len(sample_df),
    )
    logger.info("Explainability workflow complete. Top global feature: %s", importance.iloc[0]["feature"] if not importance.empty else "n/a")


if __name__ == "__main__":
    main()
