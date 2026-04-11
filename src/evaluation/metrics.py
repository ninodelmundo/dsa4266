import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from typing import Dict, Optional, Tuple


DEFAULT_METRIC_WEIGHTS = {
    "f1": 0.5,
    "roc_auc": 0.25,
    "c_index": 0.25,
}


def compute_c_index(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Concordance index for binary classification scores.
    Compares every positive-negative pair and measures how often the
    positive example receives a higher score than the negative example.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    pos_scores = y_prob[y_true == 1]
    neg_scores = y_prob[y_true == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")

    concordant = 0.0
    total_pairs = 0
    for pos_score in pos_scores:
        comparisons = pos_score - neg_scores
        concordant += np.sum(comparisons > 0)
        concordant += 0.5 * np.sum(comparisons == 0)
        total_pairs += len(neg_scores)

    return float(concordant / total_pairs) if total_pairs else float("nan")


def compute_composite_score(
    metrics: Dict[str, float],
    metric_weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute the weighted model-selection score used in optimization."""
    weights = metric_weights or DEFAULT_METRIC_WEIGHTS
    return float(
        sum(float(weights.get(metric_name, 0.0)) * float(metrics.get(metric_name, 0.0))
            for metric_name in ["f1", "roc_auc", "c_index"])
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    metric_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute the full evaluation metric suite."""
    y_pred_thresh = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred_thresh)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_thresh, average="binary", zero_division=0
    )

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = float("nan")

    c_index = compute_c_index(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred_thresh)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "c_index": c_index,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "threshold": threshold,
    }
    if metric_weights is not None:
        metrics["composite_score"] = compute_composite_score(
            metrics, metric_weights
        )
    return metrics


def find_optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1"
) -> float:
    """Find the probability threshold that maximises recall-weighted F1."""
    best_threshold = 0.5
    best_score = 0.0
    for t in np.arange(0.1, 0.91, 0.01):
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            _, _, score, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
        elif metric == "recall":
            _, score, _, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
        if score > best_score:
            best_score = score
            best_threshold = t
    return best_threshold


def calibrate_temperature(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    model_type: str = "fast_multimodal",
) -> float:
    """
    Fit a single temperature scalar on validation logits to calibrate
    probabilities (Guo et al., 2017).  Minimises NLL on val set.
    Returns optimal temperature T (logits / T before softmax).
    """
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["label"].to(device)
            if model_type == "fast_multimodal":
                logits = model(
                    url_tokens=batch["url"].to(device),
                    url_features=batch["url_features"].to(device),
                    html_features=batch["html_features"].to(device),
                    text_emb=batch["text_emb"].to(device),
                    visual_emb=batch["visual_emb"].to(device),
                )
            elif model_type == "url":
                logits = model(url_tokens=batch["url"].to(device))
            elif model_type == "fast_text":
                logits = model(text_emb=batch["text_emb"].to(device))
            elif model_type == "fast_visual":
                logits = model(visual_emb=batch["visual_emb"].to(device))
            elif model_type == "fast_url":
                logits = model(
                    url_tokens=batch["url"].to(device),
                    url_features=batch["url_features"].to(device),
                )
            elif model_type == "fast_html":
                logits = model(html_features=batch["html_features"].to(device))
            else:
                logits = model(url_tokens=batch["url"].to(device))
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Grid search for T that minimises NLL
    best_t, best_nll = 1.0, float("inf")
    for t in np.arange(0.5, 5.01, 0.05):
        nll = torch.nn.functional.cross_entropy(all_logits / t, all_labels).item()
        if nll < best_nll:
            best_nll = nll
            best_t = t
    return round(best_t, 2)


def collect_predictions_calibrated(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    model_type: str = "fast_multimodal",
    temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like collect_predictions but applies temperature scaling to logits."""
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["label"].to(device)
            if model_type == "fast_multimodal":
                logits = model(
                    url_tokens=batch["url"].to(device),
                    url_features=batch["url_features"].to(device),
                    html_features=batch["html_features"].to(device),
                    text_emb=batch["text_emb"].to(device),
                    visual_emb=batch["visual_emb"].to(device),
                )
            elif model_type == "url":
                logits = model(url_tokens=batch["url"].to(device))
            elif model_type == "fast_text":
                logits = model(text_emb=batch["text_emb"].to(device))
            elif model_type == "fast_visual":
                logits = model(visual_emb=batch["visual_emb"].to(device))
            elif model_type == "fast_url":
                logits = model(
                    url_tokens=batch["url"].to(device),
                    url_features=batch["url_features"].to(device),
                )
            elif model_type == "fast_html":
                logits = model(html_features=batch["html_features"].to(device))
            else:
                logits = model(url_tokens=batch["url"].to(device))

            probs = torch.softmax(logits / temperature, dim=-1)[:, 1].cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob


def collect_predictions(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    model_type: str = "multimodal",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and collect (y_true, y_pred, y_prob)."""
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["label"].to(device)

            if model_type == "multimodal":
                logits = model(
                    url_tokens=batch["url"].to(device),
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    images=batch["image"].to(device),
                )
            elif model_type == "url":
                logits = model(url_tokens=batch["url"].to(device))
            elif model_type == "text":
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
            elif model_type == "visual":
                logits = model(images=batch["image"].to(device))
            elif model_type == "fast_multimodal":
                logits = model(
                    url_tokens=batch["url"].to(device),
                    url_features=batch["url_features"].to(device),
                    html_features=batch["html_features"].to(device),
                    text_emb=batch["text_emb"].to(device),
                    visual_emb=batch["visual_emb"].to(device),
                )
            elif model_type == "fast_text":
                logits = model(text_emb=batch["text_emb"].to(device))
            elif model_type == "fast_visual":
                logits = model(visual_emb=batch["visual_emb"].to(device))
            elif model_type == "fast_url":
                logits = model(
                    url_tokens=batch["url"].to(device),
                    url_features=batch["url_features"].to(device),
                )
            elif model_type == "fast_html":
                logits = model(html_features=batch["html_features"].to(device))

            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob
