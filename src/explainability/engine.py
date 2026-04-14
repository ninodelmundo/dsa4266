"""Core explainability workflow for the fast cached-feature pipeline."""

from __future__ import annotations

import html
import json
import math
import re
import shutil
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.data.data_utils import (
    HTML_FEATURES_DIM,
    URL_FEATURES_DIM,
    clean_html_text,
    extract_url_features,
    get_image_transforms,
    url_to_feature_tensor,
    url_to_tensor,
)
from src.experiments.common import apply_overrides, clone_config
from src.experiments.tuning import build_fusion_model, build_unimodal_model
from src.models.fusion_model import FastFusionClassifier
from src.models.url_model import build_url_encoder

from .constants import (
    DASHBOARD_EXCLUDE_COLUMNS,
    MODALITIES,
)


@dataclass
class LoadedModel:
    """A model with the metadata needed for calibrated predictions."""

    model: torch.nn.Module
    model_type: str
    config: dict
    checkpoint_path: Optional[str]
    temperature: float = 1.0
    threshold: float = 0.5


class LegacyFastFusionClassifier(FastFusionClassifier):
    """
    Compatibility wrapper for checkpoints produced by the earlier fast-fusion
    projection heads. This is intentionally scoped to explainability loading so
    current training code can keep using the canonical FastFusionClassifier.
    """

    def __init__(self, config: dict):
        torch.nn.Module.__init__(self)
        fusion_cfg = config["fusion"]
        self.strategy = fusion_cfg["strategy"]
        self.projected_dim = fusion_cfg["projected_dim"]
        self.hidden_dim = fusion_cfg["hidden_dim"]
        self.dropout_p = fusion_cfg["dropout"]
        self.disabled_modalities = set(fusion_cfg.get("disabled_modalities", []))
        self.use_url_scalar_features = fusion_cfg.get("use_url_scalar_features", True)
        self.active_modalities = [
            name
            for name in ["url", "text", "visual", "html"]
            if name not in self.disabled_modalities
        ]
        if not self.active_modalities:
            raise ValueError("LegacyFastFusionClassifier requires at least one active modality")

        self.url_encoder = build_url_encoder(config)
        url_input_dim = config["url"]["output_dim"] + (
            URL_FEATURES_DIM if self.use_url_scalar_features else 0
        )
        text_hidden = int(config["text"].get("classifier_hidden_dim", 512))
        visual_hidden = int(config["visual"].get("classifier_hidden_dim", 512))
        html_hidden = int(config.get("html", {}).get("classifier_bottleneck_dim", 64))

        self.url_proj = torch.nn.Sequential(
            torch.nn.Linear(url_input_dim, self.projected_dim),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.projected_dim),
        )
        self.text_proj = torch.nn.Sequential(
            torch.nn.Linear(768, text_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_p),
            torch.nn.Linear(text_hidden, self.projected_dim),
            torch.nn.LayerNorm(self.projected_dim),
        )
        self.visual_proj = torch.nn.Sequential(
            torch.nn.Dropout(self.dropout_p),
            torch.nn.Linear(1280, visual_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_p),
            torch.nn.Linear(visual_hidden, self.projected_dim),
            torch.nn.LayerNorm(self.projected_dim),
        )
        self.html_proj = torch.nn.Sequential(
            torch.nn.Linear(HTML_FEATURES_DIM, html_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(html_hidden, self.projected_dim),
            torch.nn.LayerNorm(self.projected_dim),
        )

        if self.strategy == "weighted":
            self.modality_weights = torch.nn.Parameter(torch.ones(len(self.active_modalities)))
        elif self.strategy == "attention":
            self.attention = torch.nn.MultiheadAttention(
                embed_dim=self.projected_dim,
                num_heads=fusion_cfg.get("attention_heads", 4),
                dropout=self.dropout_p,
                batch_first=True,
            )
            self.attention_norm = torch.nn.LayerNorm(self.projected_dim)

        classifier_in = (
            self.projected_dim * len(self.active_modalities)
            if self.strategy == "concatenation"
            else self.projected_dim
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(classifier_in, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_p),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_p),
            torch.nn.Linear(self.hidden_dim // 2, 2),
        )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_output_dir(path: Path) -> Path:
    """Clear generated explainability artifacts so each run is self-consistent."""
    ensure_dir(path)
    generated_files = [
        "dashboard_inputs.csv",
        "dashboard_predictions.csv",
        "explainability_report.html",
        "fusion_modality_contributions.csv",
        "fusion_modality_contributions.png",
        "global_importance.csv",
        "global_importance.png",
        "local_modality_contributions.csv",
        "projected_vector_diagnostic_note.txt",
        "projected_vector_modality_diagnostic.csv",
        "projected_vector_modality_diagnostic.png",
        "README.md",
        "shapash_explainer.pkl",
        "shapash_dashboard.html",
        "summary.json",
        "surrogate_fidelity.json",
    ]
    for filename in generated_files:
        file_path = path / filename
        if file_path.exists():
            file_path.unlink()
    local_dir = path / "local"
    if local_dir.exists():
        shutil.rmtree(local_dir)
    return path


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Optional[str], device, logger) -> bool:
    if not checkpoint_path:
        return False
    path = Path(checkpoint_path)
    if not path.exists():
        logger.warning("Checkpoint not found: %s", path)
        return False
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        logger.warning("Could not load checkpoint %s: %s", path, exc)
        return False
    logger.info("Loaded checkpoint: %s", path)
    return True


def _looks_like_legacy_fusion_checkpoint(checkpoint_path: Optional[str], device) -> bool:
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return False
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    except Exception:
        return False
    return (
        "classifier.6.weight" in state_dict
        or "visual_proj.4.weight" in state_dict
        or "text_proj.3.weight" in state_dict
    )


def _apply_best_unimodal_params(config: dict, modality: str, params: dict) -> dict:
    """Mirror optimization parameter placement so checkpoints match model shapes."""
    config = clone_config(config)
    for key, value in params.items():
        if key in {
            "batch_size",
            "optimizer",
            "scheduler",
            "sampling_strategy",
            "class_weights",
            "learning_rate",
            "weight_decay",
        }:
            config["training"][key] = value
        elif key == "fusion_dropout":
            config["fusion"]["dropout"] = float(value)
        elif modality == "url" and key in {
            "model_type",
            "embedding_dim",
            "hidden_dim",
            "dropout",
            "classifier_hidden_dim",
            "classifier_bottleneck_dim",
            "use_url_scalar_features",
        }:
            config["url"][key] = value
        elif modality == "text" and key in {
            "dropout",
            "classifier_hidden_dim",
            "classifier_bottleneck_dim",
        }:
            config["text"][key] = value
        elif modality == "visual" and key in {
            "dropout",
            "classifier_hidden_dim",
            "classifier_bottleneck_dim",
        }:
            config["visual"][key] = value
        elif modality == "html" and key in {
            "classifier_hidden_dim",
            "classifier_bottleneck_dim",
        }:
            config["html"][key] = value
    return config


def _fusion_checkpoint_candidates(config: dict) -> List[Tuple[dict, Optional[str], str]]:
    """Return fast-fusion checkpoint candidates in the configured priority order."""
    output_dir = Path(config["project"]["output_dir"])
    explain_cfg = config.get("explainability", {})
    configured = explain_cfg.get("checkpoints", {}).get("fusion")
    best_overrides = output_dir / "optimization" / "fusion" / "best_fusion_overrides.json"

    candidate_paths = []
    if configured:
        candidate_paths.append((Path(configured), "configured"))

    ablation_summary = _read_json(output_dir / "optimization" / "ablation" / "summary.json")
    full_tuned = ablation_summary.get("full_tuned", {})
    if full_tuned.get("checkpoint_path"):
        candidate_paths.append((Path(full_tuned["checkpoint_path"]), "ablation_full_tuned"))

    fusion_summary = _read_json(output_dir / "optimization" / "fusion" / "summary.json")
    best_record = fusion_summary.get("best_record", {})
    if best_record.get("checkpoint_path"):
        candidate_paths.append((Path(best_record["checkpoint_path"]), "fusion_optimization_best"))

<<<<<<< HEAD
    baseline_fusion_summary = _read_json(output_dir / "baseline" / "fusion" / "summary.json")
    baseline_fusion_record = baseline_fusion_summary.get("best_record", {})
    if baseline_fusion_record.get("checkpoint_path"):
        candidate_paths.append(
            (Path(baseline_fusion_record["checkpoint_path"]), "baseline_fusion_standardised")
        )

    candidate_paths.append((output_dir / "baseline" / "fusion" / "best_model.pt", "baseline_fusion_legacy_path"))
=======
>>>>>>> origin/main
    candidate_paths.append((output_dir / "multimodal" / "best_model.pt", "baseline_multimodal"))

    candidates = []
    for path, source in candidate_paths:
        if not path.exists():
            continue
        model_config = clone_config(config)
        if source in {"ablation_full_tuned", "fusion_optimization_best", "configured"} and best_overrides.exists():
            apply_overrides(model_config, _read_json(best_overrides))
        candidates.append((model_config, str(path), source))
    if not candidates:
        candidates.append((clone_config(config), None, "untrained_default"))
    return candidates


def load_fusion_model(config: dict, device, logger) -> LoadedModel:
    for model_config, checkpoint_path, source in _fusion_checkpoint_candidates(config):
        if _looks_like_legacy_fusion_checkpoint(checkpoint_path, device):
            legacy_model = LegacyFastFusionClassifier(model_config)
            if _load_checkpoint(legacy_model, checkpoint_path, device, logger):
                legacy_model.to(device).eval()
                logger.info("Fusion checkpoint source: %s (legacy-compatible architecture)", source)
                return LoadedModel(legacy_model, "fast_multimodal", model_config, checkpoint_path)

        model, model_type, _ = build_fusion_model(model_config)
        if _load_checkpoint(model, checkpoint_path, device, logger):
            model.to(device).eval()
            logger.info("Fusion checkpoint source: %s", source)
            return LoadedModel(model, model_type, model_config, checkpoint_path)

        legacy_model = LegacyFastFusionClassifier(model_config)
        if _load_checkpoint(legacy_model, checkpoint_path, device, logger):
            legacy_model.to(device).eval()
            logger.info("Fusion checkpoint source: %s (legacy-compatible architecture)", source)
            return LoadedModel(legacy_model, "fast_multimodal", model_config, checkpoint_path)

    model_config = clone_config(config)
    model, model_type, _ = build_fusion_model(model_config)
    logger.warning("Using an untrained/default fusion model for explainability.")
    model.to(device).eval()
    return LoadedModel(model, model_type, model_config, None)


<<<<<<< HEAD
def _load_modality_summary(output_dir: Path, modality: str) -> Dict:
    """Prefer the Optuna-tuned unimodal summary; fall back to the standardised baseline."""
    optuna_aggregate = _read_json(output_dir / "optimization" / "unimodal" / "summary.json")
    if optuna_aggregate and optuna_aggregate.get(modality):
        return optuna_aggregate[modality]
    optuna_modality = _read_json(output_dir / "optimization" / "unimodal" / modality / "summary.json")
    if optuna_modality:
        return optuna_modality
    baseline_modality = _read_json(output_dir / "baseline" / "unimodal" / modality / "summary.json")
    if baseline_modality:
        return baseline_modality
    return {}


def load_unimodal_models(config: dict, device, logger) -> Dict[str, LoadedModel]:
    output_dir = Path(config["project"]["output_dir"])
    loaded = {}
    for modality in MODALITIES:
        modality_summary = _load_modality_summary(output_dir, modality)
=======
def load_unimodal_models(config: dict, device, logger) -> Dict[str, LoadedModel]:
    output_dir = Path(config["project"]["output_dir"])
    summary = _read_json(output_dir / "optimization" / "unimodal" / "summary.json")
    loaded = {}
    for modality in MODALITIES:
        modality_summary = summary.get(modality, {})
>>>>>>> origin/main
        params = modality_summary.get("best_params", {})
        record = modality_summary.get("best_record", {})
        checkpoint_path = record.get("checkpoint_path")
        model_config = _apply_best_unimodal_params(config, modality, params)
        model, model_type, _ = build_unimodal_model(modality, model_config)
        if not _load_checkpoint(model, checkpoint_path, device, logger):
            logger.warning(
                "No optimized %s checkpoint was loaded; default initialized model will be used.",
                modality,
            )
            checkpoint_path = None
        model.to(device).eval()
        loaded[modality] = LoadedModel(
            model=model,
            model_type=model_type,
            config=model_config,
            checkpoint_path=checkpoint_path,
            temperature=float(record.get("calibration_temperature", 1.0) or 1.0),
            threshold=float(record.get("optimal_threshold", 0.5) or 0.5),
        )
    return loaded


def select_split_samples(
    df: pd.DataFrame,
    features: dict,
    split: str,
    max_samples: Optional[int],
    seed: int,
) -> Tuple[pd.DataFrame, dict]:
    work_df = df.copy().reset_index(drop=True)
    work_df["feature_index"] = np.arange(len(work_df))
    if "split" in work_df.columns and split != "all":
        work_df = work_df[work_df["split"] == split]
    if max_samples and len(work_df) > max_samples:
        work_df = work_df.sample(n=max_samples, random_state=seed)
    work_df = work_df.sort_values("feature_index").reset_index(drop=True)
    indices = torch.as_tensor(work_df["feature_index"].values, dtype=torch.long)
    selected_features = {
        key: value[indices] if torch.is_tensor(value) else value
        for key, value in features.items()
    }
    return work_df, selected_features


def _to_device_batch(selected_features: dict, start: int, end: int, device) -> dict:
    return {
        "url_tokens": selected_features["url_tensors"][start:end].to(device),
        "url_features": selected_features["url_features"][start:end].to(device),
        "html_features": selected_features["html_features"][start:end].to(device),
        "text_emb": selected_features["text_embeddings"][start:end].to(device),
        "visual_emb": selected_features["visual_embeddings"][start:end].to(device),
    }


def _phishing_prob(logits: torch.Tensor, temperature: float = 1.0) -> np.ndarray:
    return torch.softmax(logits / max(float(temperature), 1e-6), dim=-1)[:, 1].detach().cpu().numpy()


def predict_loaded_model(
    loaded: LoadedModel,
    selected_features: dict,
    device,
    batch_size: int = 128,
) -> np.ndarray:
    probs = []
    n = len(selected_features["labels"])
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = _to_device_batch(selected_features, start, end, device)
            if loaded.model_type == "fast_multimodal":
                logits = loaded.model(**batch)
            elif loaded.model_type == "fast_url":
                logits = loaded.model(
                    url_tokens=batch["url_tokens"],
                    url_features=batch["url_features"],
                )
            elif loaded.model_type == "fast_text":
                logits = loaded.model(text_emb=batch["text_emb"])
            elif loaded.model_type == "fast_visual":
                logits = loaded.model(visual_emb=batch["visual_emb"])
            elif loaded.model_type == "fast_html":
                logits = loaded.model(html_features=batch["html_features"])
            else:
                raise ValueError(f"Unsupported model type for explainability: {loaded.model_type}")
            probs.append(_phishing_prob(logits, loaded.temperature))
    return np.concatenate(probs)


def _html_feature_dict(html_content: str) -> dict:
    html = str(html_content) if html_content else ""
    html_lower = html.lower()
    all_hrefs = re.findall(r"href\s*=", html_lower)
    ext_hrefs = re.findall(r"href\s*=\s*[\"']https?://", html_lower)
    text_only = re.sub(r"<[^>]+>", " ", html)
    return {
        "html_form_count": html_lower.count("<form"),
        "html_input_count": html_lower.count("<input"),
        "html_has_password": int("password" in html_lower and "<input" in html_lower),
        "html_script_count": html_lower.count("<script"),
        "html_iframe_count": html_lower.count("<iframe"),
        "html_meta_refresh": int("http-equiv" in html_lower and "refresh" in html_lower),
        "html_external_link_ratio": len(ext_hrefs) / max(len(all_hrefs), 1),
        "html_visible_text_length": len(text_only.split()),
    }


def _agreement_entropy(values: Iterable[float]) -> float:
    arr = np.clip(np.asarray(list(values), dtype=float), 1e-8, 1.0)
    arr = arr / arr.sum()
    return float(-(arr * np.log2(arr)).sum())


def build_dashboard_frame(
    sample_df: pd.DataFrame,
    selected_features: dict,
    fusion_probs: np.ndarray,
    unimodal_probs: Dict[str, np.ndarray],
    threshold: float = 0.5,
) -> pd.DataFrame:
    records = []
    labels = selected_features["labels"].cpu().numpy()
    for row_idx, row in sample_df.reset_index(drop=True).iterrows():
        url = str(row.get("url", ""))
        url_raw = extract_url_features(url)
        url_record = {
            "url_length": url_raw["length"],
            "url_num_dots": url_raw["num_dots"],
            "url_num_hyphens": url_raw["num_hyphens"],
            "url_num_slashes": url_raw["num_slashes"],
            "url_num_digits": url_raw["num_digits"],
            "url_num_special": url_raw["num_special"],
            "url_has_ip": int(url_raw["has_ip"]),
            "url_has_https": int(url_raw["has_https"]),
            "url_subdomain_count": url_raw["subdomain_count"],
        }
        html_record = _html_feature_dict(str(row.get("html_content", "")))
        modality_values = [float(unimodal_probs[name][row_idx]) for name in MODALITIES]
        record = {
            "sample_id": int(row_idx),
            "feature_index": int(row["feature_index"]),
            "split": row.get("split", ""),
            "url": url,
            "label": int(labels[row_idx]),
            "predicted_label": int(fusion_probs[row_idx] >= threshold),
            "fusion_phishing_prob": float(fusion_probs[row_idx]),
            "confidence": float(max(fusion_probs[row_idx], 1.0 - fusion_probs[row_idx])),
            "url_phishing_prob": float(unimodal_probs["url"][row_idx]),
            "text_phishing_prob": float(unimodal_probs["text"][row_idx]),
            "visual_phishing_prob": float(unimodal_probs["visual"][row_idx]),
            "html_structural_phishing_prob": float(unimodal_probs["html"][row_idx]),
            "max_unimodal_prob": float(max(modality_values)),
            "unimodal_prob_range": float(max(modality_values) - min(modality_values)),
            "num_modalities_above_threshold": int(sum(value >= threshold for value in modality_values)),
            "disagreement_entropy": _agreement_entropy(modality_values),
            **url_record,
            **html_record,
        }
        records.append(record)
    return pd.DataFrame(records)


def dashboard_feature_columns(frame: pd.DataFrame) -> List[str]:
    return [
        column
        for column in frame.columns
        if column not in DASHBOARD_EXCLUDE_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]


def fit_score_surrogate(frame: pd.DataFrame, seed: int) -> Tuple[RandomForestRegressor, pd.DataFrame, np.ndarray, dict]:
    feature_columns = dashboard_feature_columns(frame)
    x = frame[feature_columns].fillna(0.0)
    y = frame["fusion_phishing_prob"].astype(float).values
    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=5,
        min_samples_leaf=2,
        random_state=seed,
    )
    model.fit(x, y)
    pred = model.predict(x)
    fidelity = {
        "n_samples": int(len(frame)),
        "n_features": int(len(feature_columns)),
        "r2": float(r2_score(y, pred)) if len(frame) > 1 else float("nan"),
        "mae": float(mean_absolute_error(y, pred)),
        "target": "fusion_phishing_prob",
        "note": "Score-level SHAP explains an interpretable surrogate of the fusion model.",
    }
    return model, x, pred, fidelity


def compute_surrogate_shap(
    surrogate: RandomForestRegressor,
    x: pd.DataFrame,
    output_dir: Path,
    logger,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    try:
        import shap

        explainer = shap.TreeExplainer(surrogate)
        shap_values = explainer.shap_values(x)
        values = np.asarray(shap_values)
        importance = pd.DataFrame(
            {
                "feature": x.columns,
                "mean_abs_shap": np.abs(values).mean(axis=0),
                "mean_shap": values.mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
    except Exception as exc:
        logger.warning("SHAP surrogate explanation failed; using RF feature importances: %s", exc)
        values = None
        importance = pd.DataFrame(
            {
                "feature": x.columns,
                "mean_abs_shap": surrogate.feature_importances_,
                "mean_shap": np.zeros(len(x.columns)),
            }
        ).sort_values("mean_abs_shap", ascending=False)

    importance.to_csv(output_dir / "global_importance.csv", index=False)
    _plot_importance(
        importance.head(20),
        output_dir / "global_importance.png",
        "Global Surrogate SHAP Importance",
    )
    return importance, values


def _plot_importance(df: pd.DataFrame, path: Path, title: str):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(df))))
    plot_df = df.iloc[::-1]
    ax.barh(plot_df["feature"], plot_df["mean_abs_shap"], color="#2f6f73")
    ax.set_title(title)
    ax.set_xlabel("Mean absolute contribution")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def export_shapash_explainer(
    surrogate: RandomForestRegressor,
    x: pd.DataFrame,
    y_pred: np.ndarray,
    output_dir: Path,
    logger,
) -> Optional[str]:
    try:
        from shapash import SmartExplainer

        xpl = SmartExplainer(model=surrogate)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xpl.compile(x=x, y_pred=pd.DataFrame({"fusion_phishing_prob": y_pred}))
        explainer_path = output_dir / "shapash_explainer.pkl"
        if hasattr(xpl, "save"):
            xpl.save(str(explainer_path))
            return str(explainer_path)
        logger.warning("Shapash is installed but this version does not expose SmartExplainer.save().")
    except ImportError as exc:
        logger.warning(
            "Shapash explainer export skipped because shapash is not importable. "
            "Install it in this Python environment with: python3 -m pip install shapash dash"
        )
        logger.warning(
            "Verify the install with: python3 -c \"import shapash; print(shapash.__version__)\""
        )
    except Exception as exc:
        logger.warning("Shapash explainer export skipped: %s", exc)
    return None


def _active_modalities(model: FastFusionClassifier) -> List[str]:
    if hasattr(model, "get_active_modalities"):
        return model.get_active_modalities()
    return [name for name in MODALITIES if name not in getattr(model, "disabled_modalities", set())]


def project_fast_modalities(
    model: FastFusionClassifier,
    batch: dict,
) -> Dict[str, torch.Tensor]:
    projected = {}
    if "url" not in model.disabled_modalities:
        url_encoder_out = model.url_encoder(batch["url_tokens"])
        if model.use_url_scalar_features:
            url_encoder_out = torch.cat([url_encoder_out, batch["url_features"]], dim=-1)
        projected["url"] = model.url_proj(url_encoder_out)
    if "text" not in model.disabled_modalities:
        projected["text"] = model.text_proj(batch["text_emb"])
    if "visual" not in model.disabled_modalities:
        projected["visual"] = model.visual_proj(batch["visual_emb"])
    if "html" not in model.disabled_modalities:
        projected["html"] = model.html_proj(batch["html_features"])
    return projected


def fusion_from_projected(
    model: FastFusionClassifier,
    projected: Dict[str, torch.Tensor],
    active_modalities: List[str],
) -> torch.Tensor:
    embeddings = [projected[name] for name in active_modalities]
    if model.strategy == "concatenation":
        fused = torch.cat(embeddings, dim=-1)
    elif model.strategy == "weighted":
        weights = torch.softmax(model.modality_weights, dim=0)
        fused = sum(weight * emb for weight, emb in zip(weights, embeddings))
    elif model.strategy == "attention":
        stack = torch.stack(embeddings, dim=1)
        attended, _ = model.attention(stack, stack, stack)
        attended = model.attention_norm(attended + stack)
        fused = attended.mean(dim=1)
    else:
        raise ValueError(f"Unknown fusion strategy: {model.strategy}")
    return model.classifier(fused)


def _coalition_probability(
    model: FastFusionClassifier,
    sample_projected: Dict[str, torch.Tensor],
    background_projected: Dict[str, torch.Tensor],
    active_modalities: List[str],
    coalition: Iterable[str],
) -> float:
    coalition = set(coalition)
    merged = {
        name: sample_projected[name] if name in coalition else background_projected[name]
        for name in active_modalities
    }
    logits = fusion_from_projected(model, merged, active_modalities)
    return float(torch.softmax(logits, dim=-1)[:, 1].detach().cpu().item())


def compute_modality_shapley(
    fusion: LoadedModel,
    selected_features: dict,
    device,
    background_samples: int,
    local_samples: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model = fusion.model
    active = _active_modalities(model)
    n = min(local_samples, len(selected_features["labels"]))
    bg_n = min(background_samples, len(selected_features["labels"]))
    with torch.no_grad():
        bg_batch = _to_device_batch(selected_features, 0, bg_n, device)
        bg_proj_full = project_fast_modalities(model, bg_batch)
        background_projected = {
            name: tensor.mean(dim=0, keepdim=True)
            for name, tensor in bg_proj_full.items()
        }

    local_records = []
    m = len(active)
    factorial_m = math.factorial(m)
    for sample_id in range(n):
        with torch.no_grad():
            sample_batch = _to_device_batch(selected_features, sample_id, sample_id + 1, device)
            sample_projected = project_fast_modalities(model, sample_batch)
            empty_prob = _coalition_probability(model, sample_projected, background_projected, active, [])
            full_prob = _coalition_probability(model, sample_projected, background_projected, active, active)

        values_by_subset = {}
        for size in range(m + 1):
            for subset in combinations(active, size):
                values_by_subset[frozenset(subset)] = _coalition_probability(
                    model,
                    sample_projected,
                    background_projected,
                    active,
                    subset,
                )

        for modality in active:
            phi = 0.0
            others = [name for name in active if name != modality]
            for size in range(m):
                for subset in combinations(others, size):
                    subset_key = frozenset(subset)
                    with_key = frozenset((*subset, modality))
                    weight = math.factorial(size) * math.factorial(m - size - 1) / factorial_m
                    phi += weight * (values_by_subset[with_key] - values_by_subset[subset_key])
            local_records.append(
                {
                    "sample_id": sample_id,
                    "modality": modality,
                    "shapley_value": float(phi),
                    "abs_shapley_value": float(abs(phi)),
                    "empty_probability": empty_prob,
                    "full_probability": full_prob,
                }
            )

    local_df = pd.DataFrame(local_records)
    global_df = (
        local_df.groupby("modality", as_index=False)
        .agg(
            mean_shapley_value=("shapley_value", "mean"),
            mean_abs_shapley_value=("abs_shapley_value", "mean"),
        )
        .sort_values("mean_abs_shapley_value", ascending=False)
    )
    return local_df, global_df


def save_modality_contributions(local_df: pd.DataFrame, global_df: pd.DataFrame, output_dir: Path):
    local_df.to_csv(output_dir / "local_modality_contributions.csv", index=False)
    global_df.to_csv(output_dir / "fusion_modality_contributions.csv", index=False)
    global_df.to_csv(output_dir / "projected_vector_modality_diagnostic.csv", index=False)
    (output_dir / "projected_vector_diagnostic_note.txt").write_text(
        "Projected-vector diagnostics aggregate contribution at the modality level. "
        "Individual projected dimensions are intentionally not presented as human-readable features.\n"
    )
    if not global_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(global_df["modality"], global_df["mean_abs_shapley_value"], color="#b46b41")
        ax.set_title("Fusion Modality Contributions")
        ax.set_ylabel("Mean absolute Shapley value")
        plt.tight_layout()
        fig.savefig(output_dir / "fusion_modality_contributions.png", dpi=150, bbox_inches="tight")
        fig.savefig(output_dir / "projected_vector_modality_diagnostic.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _url_segments(url: str) -> Dict[str, str]:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    netloc_parts = parsed.netloc.split(".")
    domain = ".".join(netloc_parts[-2:]) if len(netloc_parts) >= 2 else parsed.netloc
    subdomain = ".".join(netloc_parts[:-2]) if len(netloc_parts) > 2 else ""
    return {
        "protocol": parsed.scheme,
        "subdomain": subdomain,
        "domain": domain,
        "path": parsed.path,
        "query": parsed.query,
    }


def url_segment_contributions(
    loaded: LoadedModel,
    url: str,
    device,
) -> List[dict]:
    max_length = loaded.config["url"]["max_length"]

    def predict_url(candidate: str) -> float:
        tokens = url_to_tensor(candidate, max_length=max_length).unsqueeze(0).to(device)
        features = url_to_feature_tensor(candidate).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = loaded.model(url_tokens=tokens, url_features=features)
        return float(_phishing_prob(logits, loaded.temperature)[0])

    base = predict_url(url)
    records = []
    for name, value in _url_segments(url).items():
        if not value:
            continue
        perturbed = url.replace(value, "", 1)
        perturbed_prob = predict_url(perturbed)
        records.append(
            {
                "segment": name,
                "value": value[:120],
                "base_probability": base,
                "perturbed_probability": perturbed_prob,
                "contribution": base - perturbed_prob,
            }
        )
    return sorted(records, key=lambda row: abs(row["contribution"]), reverse=True)


def _write_url_highlight(records: List[dict], url: str, path: Path):
    rows = "\n".join(
        f"<tr><td>{r['segment']}</td><td>{r['value']}</td><td>{r['contribution']:.4f}</td></tr>"
        for r in records
    )
    path.write_text(
        "<html><body><h2>URL Segment Contributions</h2>"
        f"<p><code>{url}</code></p>"
        "<table border='1' cellpadding='6'><tr><th>Segment</th><th>Value</th><th>Contribution</th></tr>"
        f"{rows}</table></body></html>"
    )


def _mean_pool_text_embeddings(texts: List[str], config: dict, device) -> Optional[torch.Tensor]:
    try:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config["text"]["model_name"])
        encoder = AutoModel.from_pretrained(config["text"]["model_name"]).to(device).eval()
        encodings = tokenizer(
            texts,
            max_length=config["text"]["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ids = encodings["input_ids"].to(device)
        mask = encodings["attention_mask"].to(device)
        with torch.no_grad():
            hidden = encoder(input_ids=ids, attention_mask=mask).last_hidden_state
            mask_expanded = mask.unsqueeze(-1).float()
            embeddings = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return embeddings
    except Exception:
        return None


def text_word_contributions(
    loaded: LoadedModel,
    text: str,
    device,
    max_words: int = 80,
) -> List[dict]:
    words = re.findall(r"\S+", text)[:max_words]
    if not words:
        return []
    candidates = [" ".join(words)]
    candidates.extend(" ".join(words[:idx] + words[idx + 1 :]) for idx in range(len(words)))
    embeddings = _mean_pool_text_embeddings(candidates, loaded.config, device)
    if embeddings is None:
        return []
    with torch.no_grad():
        probs = _phishing_prob(loaded.model(text_emb=embeddings), loaded.temperature)
    base = float(probs[0])
    records = []
    for idx, word in enumerate(words):
        records.append(
            {
                "token": word,
                "contribution": base - float(probs[idx + 1]),
            }
        )
    return sorted(records, key=lambda row: abs(row["contribution"]), reverse=True)


def _write_text_highlight(records: List[dict], path: Path):
    rows = "\n".join(
        f"<tr><td>{r['token']}</td><td>{r['contribution']:.4f}</td></tr>"
        for r in records[:40]
    )
    path.write_text(
        "<html><body><h2>Text Token Contributions</h2>"
        "<p>Positive values increase the phishing probability in the local perturbation explanation.</p>"
        "<table border='1' cellpadding='6'><tr><th>Token</th><th>Contribution</th></tr>"
        f"{rows}</table></body></html>"
    )


def _load_efficientnet_backbone(logger):
    from torchvision import models
    from torchvision.models import EfficientNet_B0_Weights

    try:
        backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    except Exception as exc:
        logger.warning("Could not load ImageNet EfficientNet weights for Grad-CAM; using random weights: %s", exc)
        backbone = models.efficientnet_b0(weights=None)
    backbone.classifier = torch.nn.Identity()
    return backbone


def save_gradcam_and_occlusion(
    loaded_visual: LoadedModel,
    image_path: str,
    output_prefix: Path,
    device,
    logger,
    occlusion_grid: int = 4,
    enable_gradcam: bool = True,
    enable_occlusion: bool = True,
) -> Dict[str, Optional[str]]:
    result = {"gradcam_path": None, "occlusion_path": None}
    if not enable_gradcam and not enable_occlusion:
        return result
    if not image_path or not Path(image_path).exists():
        return result
    try:
        backbone = _load_efficientnet_backbone(logger).to(device).eval()
        image_size = loaded_visual.config["visual"]["image_size"]
        transform = get_image_transforms(image_size, augment=False)
        pil = Image.open(image_path).convert("RGB").resize((image_size, image_size))
        image_tensor = transform(pil).unsqueeze(0).to(device)

        if enable_gradcam:
            target_layer = backbone.features[-1]
            activations = {}
            gradients = {}

            def forward_hook(_, __, output):
                activations["value"] = output

            def backward_hook(_, grad_input, grad_output):
                gradients["value"] = grad_output[0]

            handle_fwd = target_layer.register_forward_hook(forward_hook)
            handle_bwd = target_layer.register_full_backward_hook(backward_hook)
            image_tensor.requires_grad_(True)

            features = backbone(image_tensor)
            logits = loaded_visual.model(visual_emb=features)
            score = torch.softmax(logits, dim=-1)[:, 1].sum()
            backbone.zero_grad()
            loaded_visual.model.zero_grad()
            score.backward()

            weights = gradients["value"].mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activations["value"]).sum(dim=1)).squeeze()
            cam = cam.detach().cpu().numpy()
            cam = (cam - cam.min()) / max(cam.max() - cam.min(), 1e-8)

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(pil)
            ax.imshow(cam, cmap="jet", alpha=0.45, extent=(0, image_size, image_size, 0))
            ax.axis("off")
            gradcam_path = output_prefix.with_name(output_prefix.name + "_gradcam.png")
            fig.savefig(gradcam_path, dpi=150, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            result["gradcam_path"] = str(gradcam_path)

            handle_fwd.remove()
            handle_bwd.remove()

        if enable_occlusion:
            image_tensor = transform(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                base_prob = float(_phishing_prob(loaded_visual.model(visual_emb=backbone(image_tensor)), loaded_visual.temperature)[0])
                drops = np.zeros((occlusion_grid, occlusion_grid), dtype=float)
                for row in range(occlusion_grid):
                    for col in range(occlusion_grid):
                        occluded = image_tensor.detach().clone()
                        h0 = row * image_size // occlusion_grid
                        h1 = (row + 1) * image_size // occlusion_grid
                        w0 = col * image_size // occlusion_grid
                        w1 = (col + 1) * image_size // occlusion_grid
                        occluded[:, :, h0:h1, w0:w1] = 0.0
                        prob = float(_phishing_prob(loaded_visual.model(visual_emb=backbone(occluded)), loaded_visual.temperature)[0])
                        drops[row, col] = base_prob - prob

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(pil)
            ax.imshow(drops, cmap="magma", alpha=0.5, extent=(0, image_size, image_size, 0))
            ax.axis("off")
            occlusion_path = output_prefix.with_name(output_prefix.name + "_occlusion.png")
            fig.savefig(occlusion_path, dpi=150, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            result["occlusion_path"] = str(occlusion_path)
    except Exception as exc:
        logger.warning("Grad-CAM/occlusion failed for %s: %s", image_path, exc)
    return result


def save_local_explanations(
    sample_df: pd.DataFrame,
    selected_features: dict,
    dashboard_frame: pd.DataFrame,
    surrogate_x: pd.DataFrame,
    surrogate_shap_values: Optional[np.ndarray],
    modality_local: pd.DataFrame,
    loaded_unimodal: Dict[str, LoadedModel],
    output_dir: Path,
    device,
    logger,
    local_samples: int,
    methods: Optional[dict] = None,
):
    methods = methods or {}
    local_dir = ensure_dir(output_dir / "local")
    n = min(local_samples, len(sample_df))
    for sample_id in range(n):
        row = sample_df.iloc[sample_id]
        sample_dir = ensure_dir(local_dir / f"sample_{sample_id:03d}")

        feature_contribs = []
        if surrogate_shap_values is not None:
            values = surrogate_shap_values[sample_id]
            feature_contribs = sorted(
                [
                    {"feature": name, "contribution": float(value)}
                    for name, value in zip(surrogate_x.columns, values)
                ],
                key=lambda item: abs(item["contribution"]),
                reverse=True,
            )[:20]

        url_records = url_segment_contributions(loaded_unimodal["url"], str(row.get("url", "")), device)
        _write_url_highlight(url_records, str(row.get("url", "")), sample_dir / "url_segments.html")

        text = clean_html_text(str(row.get("html_content", "")))
        text_records = text_word_contributions(loaded_unimodal["text"], text, device)
        if text_records:
            _write_text_highlight(text_records, sample_dir / "text_tokens.html")

        image_artifacts = save_gradcam_and_occlusion(
            loaded_unimodal["visual"],
            str(row.get("image_path", "")),
            sample_dir / f"sample_{sample_id:03d}",
            device,
            logger,
            enable_gradcam=bool(methods.get("gradcam", True)),
            enable_occlusion=bool(methods.get("occlusion", True)),
        )

        modality_records = modality_local[modality_local["sample_id"] == sample_id].to_dict(orient="records")
        payload = {
            "sample_id": sample_id,
            "feature_index": int(row["feature_index"]),
            "url": str(row.get("url", "")),
            "label": int(dashboard_frame.loc[sample_id, "label"]),
            "fusion_phishing_prob": float(dashboard_frame.loc[sample_id, "fusion_phishing_prob"]),
            "surrogate_top_contributions": feature_contribs,
            "modality_shapley": modality_records,
            "url_segment_contributions": url_records,
            "text_token_contributions": text_records[:40],
            "image_artifacts": image_artifacts,
        }
        (sample_dir / "explanation.json").write_text(json.dumps(payload, indent=2))

        if feature_contribs:
            plot_df = pd.DataFrame(feature_contribs[:12]).iloc[::-1]
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ["#c44e52" if value < 0 else "#4c72b0" for value in plot_df["contribution"]]
            ax.barh(plot_df["feature"], plot_df["contribution"], color=colors)
            ax.axvline(0, color="black", linewidth=1)
            ax.set_title(f"Local Surrogate Contributions: sample {sample_id}")
            plt.tight_layout()
            fig.savefig(sample_dir / "local_contributions.png", dpi=150, bbox_inches="tight")
            plt.close(fig)


def save_dashboard_artifacts(frame: pd.DataFrame, x: pd.DataFrame, y_pred: np.ndarray, output_dir: Path):
    frame.to_csv(output_dir / "dashboard_inputs.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": frame["sample_id"],
            "fusion_phishing_prob": frame["fusion_phishing_prob"],
            "surrogate_fusion_phishing_prob": y_pred,
            "label": frame["label"],
            "predicted_label": frame["predicted_label"],
        }
    ).to_csv(output_dir / "dashboard_predictions.csv", index=False)


def _relative_link(path: Path, output_dir: Path) -> str:
    return html.escape(str(path.relative_to(output_dir)))


def _top_table(df: pd.DataFrame, columns: List[str], n: int = 10) -> str:
    if df.empty:
        return "<p>No rows available.</p>"
    return df[columns].head(n).to_html(index=False, escape=True, classes="data-table")


def save_explainability_readme(output_dir: Path, shapash_path: Optional[str], report_path: Path):
    shapash_line = (
        f"- Shapash explainer pickle: `{Path(shapash_path).name}`\n"
        "- Launch it with: `python3 scripts/serve_shapash_dashboard.py --explainer outputs/explainability/shapash_explainer.pkl --port 8050`\n"
        "- Then open: `http://127.0.0.1:8050`\n"
        if shapash_path
        else "- Shapash explainer pickle was not generated because `shapash` is not importable in this Python environment.\n"
        "- Install/verify with: `python3 -m pip install shapash dash` and `python3 -c \"import shapash; print(shapash.__version__)\"`\n"
    )
    (output_dir / "README.md").write_text(
        "# Explainability Outputs\n\n"
        "This folder contains generated explainability artifacts for the fast multimodal phishing pipeline.\n\n"
        "## Browser-Openable Report\n"
        f"- Static report: `{report_path.name}`\n\n"
        "## Local Sample Folders\n"
        "- `local/sample_XXX/` folders are per-example explanations from the selected dataset split.\n"
        "- `sample_id` is the row number inside this explainability run.\n"
        "- `feature_index` points back to the original cached feature row.\n"
        "- Each folder can include `explanation.json`, `local_contributions.png`, `url_segments.html`, `text_tokens.html`, Grad-CAM, and occlusion maps.\n\n"
        "## Shapash Interactive Dashboard\n"
        "- Shapash is an interactive local web app, not a static HTML export.\n"
        f"{shapash_line}\n"
    )


def save_explainability_report(
    output_dir: Path,
    dashboard_frame: pd.DataFrame,
    importance: pd.DataFrame,
    modality_global: pd.DataFrame,
    fidelity: dict,
    shapash_path: Optional[str],
    local_samples: int,
) -> str:
    report_path = output_dir / "explainability_report.html"
    local_count = min(local_samples, len(dashboard_frame))
    local_rows = []
    for sample_id in range(local_count):
        sample_dir = Path("local") / f"sample_{sample_id:03d}"
        row = dashboard_frame.iloc[sample_id]
        local_rows.append(
            {
                "sample_id": sample_id,
                "feature_index": int(row["feature_index"]),
                "label": int(row["label"]),
                "fusion_phishing_prob": round(float(row["fusion_phishing_prob"]), 4),
                "sample_folder": f"<a href='{sample_dir}/explanation.json'>{sample_dir}</a>",
            }
        )
    local_table = pd.DataFrame(local_rows).to_html(index=False, escape=False, classes="data-table")
    shapash_html = (
        "<p>A Shapash explainer pickle was generated. Start the interactive dashboard with "
        "<code>python3 scripts/serve_shapash_dashboard.py --explainer outputs/explainability/shapash_explainer.pkl --port 8050</code>, "
        "then open <code>http://127.0.0.1:8050</code>.</p>"
        if shapash_path
        else "<p><strong>Shapash explainer not generated.</strong> Install it in this Python environment with "
        "<code>python3 -m pip install shapash dash</code>, then verify with "
        "<code>python3 -c \"import shapash; print(shapash.__version__)\"</code>.</p>"
    )
    report_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Explainability Report</title>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45;color:#1f2933;}"
        "h1,h2{color:#102a43}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:24px;}"
        ".card{border:1px solid #d9e2ec;border-radius:12px;padding:18px;background:#f8fafc;}"
        ".data-table{border-collapse:collapse;width:100%;font-size:14px}.data-table th,.data-table td{border:1px solid #d9e2ec;padding:6px;text-align:left;}"
        ".data-table th{background:#eef2f7}img{max-width:100%;border:1px solid #d9e2ec;border-radius:8px;}code{background:#eef2f7;padding:2px 4px;border-radius:4px;}"
        "</style></head><body>"
        "<h1>Explainability Report</h1>"
        "<p>This static report summarizes the generated SHAP surrogate, fusion modality Shapley diagnostics, and local explanations.</p>"
        "<div class='grid'>"
        f"<div class='card'><h2>Surrogate Fidelity</h2><p><strong>R2:</strong> {fidelity.get('r2', float('nan')):.4f}</p>"
        f"<p><strong>MAE:</strong> {fidelity.get('mae', float('nan')):.4f}</p>"
        f"<p><strong>Samples:</strong> {int(fidelity.get('n_samples', len(dashboard_frame)))}</p></div>"
        "<div class='card'><h2>Shapash</h2>"
        f"{shapash_html}</div></div>"
        "<h2>Global Surrogate Importance</h2>"
        "<p>These are readable features explaining the score-level surrogate of the fusion model.</p>"
        "<p><img src='global_importance.png' alt='Global importance plot'></p>"
        f"{_top_table(importance, ['feature', 'mean_abs_shap', 'mean_shap'])}"
        "<h2>Fusion Modality Contributions</h2>"
        "<p>These modality-level Shapley values are computed from the real fast fusion model using projected modality groups.</p>"
        "<p><img src='fusion_modality_contributions.png' alt='Fusion modality contributions'></p>"
        f"{_top_table(modality_global, ['modality', 'mean_shapley_value', 'mean_abs_shapley_value'])}"
        "<h2>Local Samples</h2>"
        "<p>Each local folder explains one selected sample from the configured split.</p>"
        f"{local_table}"
        "<h2>Generated Files</h2>"
        "<ul>"
        "<li><a href='dashboard_inputs.csv'>dashboard_inputs.csv</a></li>"
        "<li><a href='dashboard_predictions.csv'>dashboard_predictions.csv</a></li>"
        "<li><a href='global_importance.csv'>global_importance.csv</a></li>"
        "<li><a href='fusion_modality_contributions.csv'>fusion_modality_contributions.csv</a></li>"
        "<li><a href='local_modality_contributions.csv'>local_modality_contributions.csv</a></li>"
        "</ul>"
        "</body></html>"
    )
    save_explainability_readme(output_dir, shapash_path, report_path)
    return str(report_path)


def save_summary(
    output_dir: Path,
    fusion: LoadedModel,
    unimodal: Dict[str, LoadedModel],
    fidelity: dict,
    shapash_path: Optional[str],
    report_path: Optional[str],
    sample_count: int,
):
    summary = {
        "sample_count": int(sample_count),
        "fusion_checkpoint": fusion.checkpoint_path,
        "unimodal_checkpoints": {
            modality: loaded.checkpoint_path for modality, loaded in unimodal.items()
        },
        "surrogate_fidelity": fidelity,
        "shapash_explainer": shapash_path,
        "static_report": report_path,
        "notes": [
            "Score-level SHAP explains an interpretable surrogate of the fusion model.",
            "Shapash uses a saved SmartExplainer pickle and must be opened with scripts/serve_shapash_dashboard.py.",
            "Fusion modality Shapley values are computed on projected modality representations of the real fast fusion model.",
            "Projected vector diagnostics are aggregated by modality; individual dimensions are not presented as human-readable factors.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "surrogate_fidelity.json").write_text(json.dumps(fidelity, indent=2))
