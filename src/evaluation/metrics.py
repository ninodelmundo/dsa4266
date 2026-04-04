import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict, Tuple


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
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

    cm = confusion_matrix(y_true, y_pred_thresh)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "threshold": threshold,
    }


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

            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob
