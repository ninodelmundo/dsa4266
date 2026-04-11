import copy
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from ..data.dataset_loader import PhishingDatasetLoader
from ..data.preprocessor import (
    compute_class_weights,
    create_fast_dataloaders,
    extract_and_save_features,
)
from ..evaluation.analysis import plot_training_curves
from ..evaluation.metrics import (
    calibrate_temperature,
    collect_predictions_calibrated,
    compute_metrics,
    find_optimal_threshold,
)
from ..training.trainer import Trainer
from ..utils.helpers import get_device, load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


def _to_serializable(value):
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def clone_config(config: dict) -> dict:
    return copy.deepcopy(config)


def apply_overrides(config: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            apply_overrides(config[key], value)
        else:
            config[key] = value
    return config


def get_metric_weights(config: dict) -> Dict[str, float]:
    optimization_cfg = config.get("optimization", {})
    return optimization_cfg.get(
        "metric_weights",
        {"f1": 0.5, "roc_auc": 0.25, "c_index": 0.25},
    )


def prepare_environment(config_path: str):
    config = load_config(config_path)
    set_seed(config["project"]["seed"])
    device = get_device(config)
    logger = setup_logging(config["project"]["output_dir"])
    return config, device, logger


def limit_dataset(df, max_samples: Optional[int], seed: int = 42):
    if not max_samples or len(df) <= max_samples:
        return df
    if "split" not in df.columns:
        return df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    split_names = [split for split in ["train", "val", "test"] if split in set(df["split"])]
    split_sizes = {split: int((df["split"] == split).sum()) for split in split_names}
    total = sum(split_sizes.values())
    counts = {
        split: max(1, round((size / total) * max_samples))
        for split, size in split_sizes.items()
    }

    while sum(counts.values()) > max_samples:
        reducible = [split for split, count in counts.items() if count > 1]
        if not reducible:
            break
        split_to_reduce = max(reducible, key=lambda split: counts[split])
        counts[split_to_reduce] -= 1

    while sum(counts.values()) < max_samples:
        split_to_increase = max(split_sizes, key=split_sizes.get)
        counts[split_to_increase] += 1

    samples = []
    for split in split_names:
        subset = df[df["split"] == split]
        take = min(len(subset), counts[split])
        samples.append(subset.sample(n=take, random_state=seed))
    return pd.concat(samples, ignore_index=True)


def load_fast_resources(config: dict):
    loader = PhishingDatasetLoader(config)
    df = loader.build_merged_dataset()
    max_samples = config["data"].get("max_samples")
    df = limit_dataset(df, max_samples, seed=config["project"]["seed"])
    features = extract_and_save_features(df, config)
    return df, features


def build_fast_loaders_and_weights(config: dict, df, features, device):
    train_loader, val_loader, test_loader = create_fast_dataloaders(
        df,
        features,
        config,
        batch_size=config["training"]["batch_size"],
        sampling_strategy=config["training"].get("sampling_strategy", "shuffle"),
    )
    class_weights = None
    if config["training"].get("class_weights", True):
        class_weights = compute_class_weights(df, device)
    return train_loader, val_loader, test_loader, class_weights


def train_and_evaluate_model(
    *,
    model,
    model_type: str,
    model_name: str,
    config: dict,
    device,
    df,
    features,
    output_dir: str,
    trial=None,
) -> Dict:
    metric_weights = get_metric_weights(config)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, class_weights = build_fast_loaders_and_weights(
        config,
        df,
        features,
        device,
    )

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
        trial=trial,
        trial_report_metric=config.get("optimization", {}).get(
            "study_metric", "composite_score"
        ),
    )
    history = trainer.fit(output_dir)
    plot_training_curves(history, output_dir, model_name)

    temperature = calibrate_temperature(model, val_loader, device, model_type)
    val_true, _, val_prob = collect_predictions_calibrated(
        model,
        val_loader,
        device,
        model_type,
        temperature=temperature,
    )
    optimal_threshold = find_optimal_threshold(val_true, val_prob)
    val_pred = (val_prob >= optimal_threshold).astype(int)
    val_metrics = compute_metrics(
        val_true,
        val_pred,
        val_prob,
        optimal_threshold,
        metric_weights=metric_weights,
    )

    test_true, _, test_prob = collect_predictions_calibrated(
        model,
        test_loader,
        device,
        model_type,
        temperature=temperature,
    )
    test_pred = (test_prob >= optimal_threshold).astype(int)
    test_metrics = compute_metrics(
        test_true,
        test_pred,
        test_prob,
        optimal_threshold,
        metric_weights=metric_weights,
    )

    results = {
        "model_name": model_name,
        "model_type": model_type,
        "checkpoint_path": str(Path(output_dir) / "best_model.pt"),
        "calibration_temperature": temperature,
        "optimal_threshold": optimal_threshold,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
    }
    if hasattr(model, "get_modality_weights"):
        weights = model.get_modality_weights()
        if weights is not None:
            results["modality_weights"] = [float(x) for x in weights]
    if hasattr(model, "get_active_modalities"):
        results["active_modalities"] = model.get_active_modalities()
    return results


def flatten_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def save_records(records: Iterable[Dict], output_dir: str, stem: str) -> Dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    serializable_records = [_to_serializable(record) for record in records]
    df = pd.DataFrame(serializable_records)
    csv_path = output_path / f"{stem}.csv"
    json_path = output_path / f"{stem}.json"
    df.to_csv(csv_path, index=False)
    with json_path.open("w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
    return {"csv": str(csv_path), "json": str(json_path)}


def save_summary(summary: Dict, output_dir: str, filename: str = "summary.json") -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / filename
    with summary_path.open("w") as f:
        json.dump(_to_serializable(summary), f, indent=2)
    return str(summary_path)


def parse_logged_baseline_metrics(log_path: str) -> Optional[Dict[str, float]]:
    path = Path(log_path)
    if not path.exists():
        return None
    pattern = re.compile(
        r"Multimodal\s+\|\s+F1:\s+(?P<f1>[0-9.]+)\s+\|\s+ROC-AUC:\s+(?P<roc_auc>[0-9.]+)"
        r"\s+\|\s+Acc:\s+(?P<accuracy>[0-9.]+)\s+\|\s+Prec:\s+(?P<precision>[0-9.]+)"
        r"\s+\|\s+Rec:\s+(?P<recall>[0-9.]+)\s+\|\s+Thresh:\s+(?P<threshold>[0-9.]+)"
    )
    for line in reversed(path.read_text().splitlines()):
        match = pattern.search(line)
        if match:
            return {key: float(value) for key, value in match.groupdict().items()}
    return None
