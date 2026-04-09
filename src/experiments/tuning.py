import copy
from typing import Dict, Tuple

from ..models.fusion_model import (
    FastFusionClassifier,
    FastHTMLOnlyClassifier,
    FastTextOnlyClassifier,
    FastURLOnlyClassifier,
    FastVisualOnlyClassifier,
)


def suggest_from_spec(trial, name: str, spec):
    if isinstance(spec, list):
        return trial.suggest_categorical(name, spec)
    if isinstance(spec, dict) and "low" in spec and "high" in spec:
        low = spec["low"]
        high = spec["high"]
        log = bool(spec.get("log", False))
        if isinstance(low, int) and isinstance(high, int):
            step = int(spec.get("step", 1))
            return trial.suggest_int(name, low, high, step=step, log=log)
        return trial.suggest_float(name, low, high, log=log)
    return spec


def apply_common_search_space(config: dict, trial) -> Dict[str, object]:
    common_space = config.get("optimization", {}).get("search_spaces", {}).get(
        "common", {}
    )
    selections = {}
    if not common_space:
        return selections

    for key, spec in common_space.items():
        trial_name = f"common.{key}"
        value = suggest_from_spec(trial, trial_name, spec)
        selections[key] = value
        if key == "batch_size":
            config["training"]["batch_size"] = int(value)
        elif key == "optimizer":
            config["training"]["optimizer"] = str(value)
        elif key == "scheduler":
            config["training"]["scheduler"] = str(value)
        elif key == "sampling_strategy":
            config["training"]["sampling_strategy"] = str(value)
        elif key == "class_weights":
            config["training"]["class_weights"] = bool(value)
        elif key == "learning_rate":
            config["training"]["learning_rate"] = float(value)
        elif key == "weight_decay":
            config["training"]["weight_decay"] = float(value)
        elif key == "fusion_dropout":
            config["fusion"]["dropout"] = float(value)

    config["training"]["checkpoint_metric"] = config["optimization"].get(
        "study_metric", "composite_score"
    )
    return selections


def apply_unimodal_search_space(config: dict, trial, modality: str) -> Dict[str, object]:
    selections = apply_common_search_space(config, trial)
    modality_space = (
        config.get("optimization", {})
        .get("search_spaces", {})
        .get(modality, {})
    )

    for key, spec in modality_space.items():
        trial_name = f"{modality}.{key}"
        value = suggest_from_spec(trial, trial_name, spec)
        selections[key] = value

        if modality == "url":
            if key in {"model_type"}:
                config["url"][key] = value
            elif key in {"embedding_dim", "hidden_dim", "classifier_hidden_dim", "classifier_bottleneck_dim"}:
                config["url"][key] = int(value)
            elif key == "use_url_scalar_features":
                config["url"][key] = bool(value)
            elif key == "dropout":
                config["url"][key] = float(value)
                config["fusion"]["dropout"] = float(value)
        elif modality == "text":
            if key in {"classifier_hidden_dim", "classifier_bottleneck_dim"}:
                config["text"][key] = int(value)
            elif key == "dropout":
                config["text"]["dropout"] = float(value)
        elif modality == "visual":
            if key in {"classifier_hidden_dim", "classifier_bottleneck_dim"}:
                config["visual"][key] = int(value)
            elif key == "dropout":
                config["visual"]["dropout"] = float(value)
        elif modality == "html":
            if key in {"classifier_hidden_dim", "classifier_bottleneck_dim"}:
                config["html"][key] = int(value)

    return selections


def apply_fusion_search_space(config: dict, trial) -> Dict[str, object]:
    selections = apply_common_search_space(config, trial)
    fusion_space = config.get("optimization", {}).get("search_spaces", {}).get(
        "fusion", {}
    )
    for key, spec in fusion_space.items():
        trial_name = f"fusion.{key}"
        value = suggest_from_spec(trial, trial_name, spec)
        selections[key] = value
        if key in {"projected_dim", "hidden_dim"}:
            config["fusion"][key] = int(value)
        elif key == "dropout":
            config["fusion"][key] = float(value)
        else:
            config["fusion"][key] = value
    config["training"]["checkpoint_metric"] = config["optimization"].get(
        "study_metric", "composite_score"
    )
    return selections


def build_unimodal_model(modality: str, config: dict):
    if modality == "url":
        return FastURLOnlyClassifier(config), "fast_url", "URL-Only"
    if modality == "text":
        return FastTextOnlyClassifier(config), "fast_text", "Text-Only"
    if modality == "visual":
        return FastVisualOnlyClassifier(config), "fast_visual", "Visual-Only"
    if modality == "html":
        return FastHTMLOnlyClassifier(config), "fast_html", "HTML-Only"
    raise ValueError(f"Unsupported modality: {modality}")


def build_fusion_model(config: dict):
    return FastFusionClassifier(config), "fast_multimodal", "Multimodal Fusion"


def build_promoted_overrides(unimodal_summaries: Dict[str, Dict]) -> Dict[str, Dict]:
    promoted = {"training": {}, "url": {}, "fusion": {}}
    url_summary = unimodal_summaries.get("url")
    if url_summary:
        best_params = url_summary.get("best_params", {})
        for key in [
            "model_type",
            "embedding_dim",
            "hidden_dim",
            "dropout",
            "classifier_hidden_dim",
            "classifier_bottleneck_dim",
            "use_url_scalar_features",
        ]:
            if key in best_params:
                promoted["url"][key] = best_params[key]

    best_overall = sorted(
        (
            (modality, summary.get("best_value", float("-inf")), summary)
            for modality, summary in unimodal_summaries.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    if best_overall:
        best_params = best_overall[0][2].get("best_params", {})
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
                promoted["training"][key] = best_params[key]
        if "fusion_dropout" in best_params:
            promoted["fusion"]["dropout"] = best_params["fusion_dropout"]

    return {section: values for section, values in promoted.items() if values}
