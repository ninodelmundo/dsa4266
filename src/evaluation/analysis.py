import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    silhouette_score,
)
from typing import Dict, List, Optional


def plot_confusion_matrix(
    y_true, y_pred, output_dir: str, title: str = "Confusion Matrix"
):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(
        output_dir, f"{title.replace(' ', '_').lower()}.png"
    )
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_roc_curves(results: Dict[str, Dict], output_dir: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, data in results.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        auc = data["metrics"]["roc_auc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Model Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_pr_curves(results: Dict[str, Dict], output_dir: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, data in results.items():
        prec, rec, _ = precision_recall_curve(data["y_true"], data["y_prob"])
        pr_auc = data["metrics"]["pr_auc"]
        ax.plot(rec, prec, label=f"{name} (PR-AUC={pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="upper right")
    plt.tight_layout()
    path = os.path.join(output_dir, "pr_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_metrics_comparison(results: Dict[str, Dict], output_dir: str):
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    model_names = list(results.keys())
    data = {
        m: [results[name]["metrics"][m] for name in model_names]
        for m in metric_keys
    }

    df = pd.DataFrame(data, index=model_names)
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison - All Metrics")
    ax.legend(loc="lower right")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = os.path.join(output_dir, "metrics_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_training_curves(
    history: Dict[str, List], output_dir: str, model_name: str
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{model_name} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_f1"], label="Train F1")
    axes[1].plot(epochs, history["val_f1"], label="Val F1")
    axes[1].set_title(f"{model_name} - F1 Score")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(
        output_dir,
        f"training_curves_{model_name.lower().replace(' ', '_')}.png",
    )
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def ablation_study_plot(ablation_results: Dict[str, Dict], output_dir: str):
    """Bar chart showing F1 drop when each modality is removed."""
    full_f1 = (
        ablation_results.get("Full Multimodal", {})
        .get("metrics", {})
        .get("f1", 0)
    )
    names, drops = [], []
    for name, data in ablation_results.items():
        f1 = data["metrics"]["f1"]
        names.append(name)
        drops.append(f1)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [
        "steelblue" if n == "Full Multimodal" else "tomato" for n in names
    ]
    bars = ax.bar(names, drops, color=colors, edgecolor="black")
    ax.axhline(y=full_f1, color="green", linestyle="--", label="Full Multimodal")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score")
    ax.set_title("Ablation Study - Modality Contribution")
    ax.legend()
    plt.xticks(rotation=20, ha="right")

    for bar, val in zip(bars, drops):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            fontsize=9,
        )
    plt.tight_layout()
    path = os.path.join(output_dir, "ablation_study.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_class_distribution(df: pd.DataFrame, output_dir: str):
    """Bar chart showing class balance (or imbalance) per split."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Overall distribution
    counts = df["label"].value_counts().sort_index()
    labels = ["Legitimate (0)", "Phishing (1)"]
    colors = ["#4CAF50", "#F44336"]
    bars = axes[0].bar(labels, counts.values, color=colors, edgecolor="black")
    axes[0].set_title("Overall Class Distribution")
    axes[0].set_ylabel("Count")
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     str(val), ha="center", fontsize=11, fontweight="bold")

    # Per-split distribution
    if "split" in df.columns:
        split_data = df.groupby(["split", "label"]).size().unstack(fill_value=0)
        split_order = ["train", "val", "test"]
        split_data = split_data.reindex([s for s in split_order if s in split_data.index])
        split_data.columns = labels
        split_data.plot(kind="bar", ax=axes[1], color=colors, edgecolor="black")
        axes[1].set_title("Class Distribution per Split")
        axes[1].set_ylabel("Count")
        axes[1].set_xlabel("")
        axes[1].legend(title="Class")
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=0)

    plt.tight_layout()
    path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_dataset_stats(df: pd.DataFrame, output_dir: str):
    """Summary statistics: URL lengths, HTML text lengths, missing data."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # URL length distribution
    url_lengths = df["url"].astype(str).str.len()
    axes[0].hist(url_lengths, bins=50, color="steelblue", edgecolor="black", alpha=0.8)
    axes[0].set_title("URL Length Distribution")
    axes[0].set_xlabel("Characters")
    axes[0].set_ylabel("Count")
    axes[0].axvline(url_lengths.median(), color="red", linestyle="--",
                    label=f"Median: {url_lengths.median():.0f}")
    axes[0].legend()

    # HTML content length distribution
    html_lengths = df["html_content"].astype(str).str.len()
    axes[1].hist(html_lengths, bins=50, color="coral", edgecolor="black", alpha=0.8)
    axes[1].set_title("HTML Content Length Distribution")
    axes[1].set_xlabel("Characters")
    axes[1].set_ylabel("Count")
    axes[1].axvline(html_lengths.median(), color="red", linestyle="--",
                    label=f"Median: {html_lengths.median():.0f}")
    axes[1].legend()

    # Missing data summary
    missing = {
        "URL": df["url"].isna().sum(),
        "HTML": (df["html_content"].isna() | (df["html_content"] == "")).sum(),
        "Image": df["image_path"].isna().sum(),
    }
    bars = axes[2].bar(missing.keys(), missing.values(), color="#FF9800", edgecolor="black")
    axes[2].set_title("Missing Data per Modality")
    axes[2].set_ylabel("Count")
    for bar, val in zip(bars, missing.values()):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(val), ha="center", fontsize=11)

    plt.tight_layout()
    path = os.path.join(output_dir, "dataset_stats.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _safe_tsne_perplexity(n_samples: int, requested: int) -> int:
    """Return a valid deterministic t-SNE perplexity for the sample size."""
    if n_samples < 4:
        return max(1, n_samples - 1)
    return max(2, min(requested, n_samples - 1, n_samples // 3))


def _compute_tsne_coords(embeddings: np.ndarray, perplexity: int = 30) -> np.ndarray:
    embeddings = np.asarray(embeddings)
    if len(embeddings) < 2:
        raise ValueError("t-SNE requires at least two samples")
    tsne = TSNE(
        n_components=2,
        perplexity=_safe_tsne_perplexity(len(embeddings), perplexity),
        random_state=42,
        max_iter=1000,
    )
    return tsne.fit_transform(embeddings)


def _plot_tsne_scatter(ax, coords: np.ndarray, labels: np.ndarray, title: str):
    for label, name, color in [(0, "Legitimate", "#4CAF50"), (1, "Phishing", "#F44336")]:
        mask = labels == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color,
            label=name,
            alpha=0.6,
            s=20,
            edgecolors="none",
        )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend()


def _silhouette_or_nan(embeddings: np.ndarray, labels: np.ndarray) -> float:
    try:
        if len(np.unique(labels)) < 2 or len(labels) < 3:
            return float("nan")
        return float(silhouette_score(embeddings, labels))
    except Exception:
        return float("nan")


def plot_embedding_tsne(features: dict, output_dir: str, perplexity: int = 30):
    """t-SNE visualization of raw cached text + visual embeddings."""
    labels = features["labels"].numpy()
    text_emb = features["text_embeddings"].numpy()
    visual_emb = features["visual_embeddings"].numpy()

    # Pre-model representation: cached DistilBERT + EfficientNet embeddings.
    combined = np.concatenate([text_emb, visual_emb], axis=1)
    coords = _compute_tsne_coords(combined, perplexity=perplexity)

    fig, ax = plt.subplots(figsize=(8, 7))
    _plot_tsne_scatter(ax, coords, labels, "t-SNE of Raw Cached Text + Visual Embeddings")
    plt.tight_layout()
    path = os.path.join(output_dir, "embedding_tsne_raw_text_visual.png")
    plt.savefig(path, dpi=150)
    plt.close()

    # Backward-compatible copy for older notebooks/reports that referenced this name.
    legacy_path = os.path.join(output_dir, "embedding_tsne.png")
    fig, ax = plt.subplots(figsize=(8, 7))
    _plot_tsne_scatter(ax, coords, labels, "t-SNE of Raw Cached Text + Visual Embeddings")
    plt.tight_layout()
    plt.savefig(legacy_path, dpi=150)
    plt.close()
    return path


def collect_raw_text_visual_embeddings(loader) -> tuple:
    """Collect held-out raw cached text+visual embeddings and labels from a dataloader."""
    raw_embeddings = []
    labels = []
    for batch in loader:
        raw_embeddings.append(
            np.concatenate(
                [
                    batch["text_emb"].detach().cpu().numpy(),
                    batch["visual_emb"].detach().cpu().numpy(),
                ],
                axis=1,
            )
        )
        labels.append(batch["label"].detach().cpu().numpy())
    return np.concatenate(raw_embeddings, axis=0), np.concatenate(labels, axis=0)


def collect_fusion_embeddings(model, loader, device) -> tuple:
    """
    Collect learned fused representations immediately before the fusion classifier.

    This mirrors FastFusionClassifier.forward up to the `fused` tensor so the
    t-SNE reflects the trained fusion representation, not raw cached features.
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    fused_embeddings = []
    labels = []
    disabled_modalities = getattr(model, "disabled_modalities", set())
    use_url_scalar_features = getattr(model, "use_url_scalar_features", True)

    with torch.no_grad():
        for batch in loader:
            url_tokens = batch["url"].to(device)
            url_features = batch["url_features"].to(device)
            html_features = batch["html_features"].to(device)
            text_emb = batch["text_emb"].to(device)
            visual_emb = batch["visual_emb"].to(device)

            projected = []
            if "url" not in disabled_modalities:
                url_encoder_out = model.url_encoder(url_tokens)
                if use_url_scalar_features:
                    url_encoder_out = torch.cat([url_encoder_out, url_features], dim=-1)
                projected.append(model.url_proj(url_encoder_out))
            if "text" not in disabled_modalities:
                projected.append(model.text_proj(text_emb))
            if "visual" not in disabled_modalities:
                projected.append(model.visual_proj(visual_emb))
            if "html" not in disabled_modalities:
                projected.append(model.html_proj(html_features))

            if model.strategy == "concatenation":
                fused = torch.cat(projected, dim=-1)
            elif model.strategy == "weighted":
                weights = F.softmax(model.modality_weights, dim=0)
                fused = sum(weight * emb for weight, emb in zip(weights, projected))
            elif model.strategy == "attention":
                stack = torch.stack(projected, dim=1)
                attended, _ = model.attention(stack, stack, stack)
                attended = model.attention_norm(attended + stack)
                fused = attended.mean(dim=1)
            else:
                raise ValueError(f"Unknown fusion strategy: {model.strategy}")

            fused_embeddings.append(fused.detach().cpu().numpy())
            labels.append(batch["label"].detach().cpu().numpy())

    return np.concatenate(fused_embeddings, axis=0), np.concatenate(labels, axis=0)


def plot_fusion_tsne_comparison(
    model,
    loader,
    output_dir: str,
    source_name: str = "optimization",
    perplexity: int = 30,
):
    """Plot raw cached feature t-SNE beside learned final fusion t-SNE."""
    raw_embeddings, raw_labels = collect_raw_text_visual_embeddings(loader)
    learned_embeddings, learned_labels = collect_fusion_embeddings(model, loader, next(model.parameters()).device)

    if not np.array_equal(raw_labels, learned_labels):
        raise ValueError("Raw and learned t-SNE labels are not aligned")

    raw_coords = _compute_tsne_coords(raw_embeddings, perplexity=perplexity)
    learned_coords = _compute_tsne_coords(learned_embeddings, perplexity=perplexity)

    fig, ax = plt.subplots(figsize=(8, 7))
    _plot_tsne_scatter(ax, raw_coords, raw_labels, "t-SNE of Raw Cached Text + Visual Embeddings")
    plt.tight_layout()
    raw_path = os.path.join(output_dir, "embedding_tsne_raw_text_visual.png")
    plt.savefig(raw_path, dpi=150)
    legacy_path = os.path.join(output_dir, "embedding_tsne.png")
    plt.savefig(legacy_path, dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 7))
    _plot_tsne_scatter(
        ax,
        learned_coords,
        learned_labels,
        f"t-SNE of Learned Final Fusion Representations ({source_name})",
    )
    plt.tight_layout()
    learned_path = os.path.join(output_dir, "embedding_tsne_final_fusion.png")
    plt.savefig(learned_path, dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _plot_tsne_scatter(axes[0], raw_coords, raw_labels, "Raw Cached Text + Visual")
    _plot_tsne_scatter(axes[1], learned_coords, learned_labels, f"Learned Fusion ({source_name})")
    fig.suptitle("t-SNE Comparison: Raw Feature Space vs Learned Fusion Space")
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "embedding_tsne_raw_vs_final_fusion.png")
    plt.savefig(comparison_path, dpi=150)
    plt.close()

    summary = {
        "source": source_name,
        "split": "test",
        "n_samples": int(len(raw_labels)),
        "raw_embedding_dim": int(raw_embeddings.shape[1]),
        "learned_fusion_dim": int(learned_embeddings.shape[1]),
        "raw_silhouette": _silhouette_or_nan(raw_embeddings, raw_labels),
        "learned_fusion_silhouette": _silhouette_or_nan(learned_embeddings, learned_labels),
        "files": {
            "raw_text_visual_tsne": raw_path,
            "legacy_raw_text_visual_tsne": legacy_path,
            "final_fusion_tsne": learned_path,
            "comparison_tsne": comparison_path,
        },
        "note": (
            "The raw plot uses cached DistilBERT text + EfficientNet visual embeddings. "
            "The learned plot uses the trained fusion vector immediately before the classifier head."
        ),
    }
    with open(os.path.join(output_dir, "embedding_tsne_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def plot_threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, output_dir: str):
    """Show how precision, recall, and F1 change across decision thresholds."""
    thresholds = np.arange(0.10, 0.91, 0.01)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    best_idx = np.argmax(f1s)
    best_t = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, label="Precision", color="#2196F3")
    ax.plot(thresholds, recalls, label="Recall", color="#FF9800")
    ax.plot(thresholds, f1s, label="F1 Score", color="#4CAF50", linewidth=2)
    ax.axvline(best_t, color="red", linestyle="--",
               label=f"Optimal Threshold: {best_t:.2f} (F1={f1s[best_idx]:.3f})")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Sweep: Precision / Recall / F1 Tradeoff")
    ax.legend()
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = os.path.join(output_dir, "threshold_sweep.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_modality_attention_weights(model, features: dict, device, output_dir: str):
    """Visualize average attention weights across the 4 modalities."""
    import torch

    model.eval()
    url_tensors = features["url_tensors"].to(device)
    url_feats = features["url_features"].to(device)
    html_feats = features["html_features"].to(device)
    text_emb = features["text_embeddings"].to(device)
    visual_emb = features["visual_embeddings"].to(device)

    # Get projected embeddings and attention weights
    with torch.no_grad():
        url_raw = torch.cat([model.url_encoder(url_tensors), url_feats], dim=-1)
        url_e = model.url_proj(url_raw)
        text_e = model.text_proj(text_emb)
        visual_e = model.visual_proj(visual_emb)
        html_e = model.html_proj(html_feats)

        stack = torch.stack([url_e, text_e, visual_e, html_e], dim=1)
        _, attn_weights = model.attention(stack, stack, stack, need_weights=True)

    # Average attention across all samples and heads: (4, 4)
    avg_attn = attn_weights.mean(dim=0).cpu().numpy()

    modalities = ["URL", "Text", "Visual", "HTML"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(avg_attn, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=modalities, yticklabels=modalities, ax=ax)
    ax.set_title("Average Cross-Modal Attention Weights")
    ax.set_xlabel("Key (attends to)")
    ax.set_ylabel("Query (from)")
    plt.tight_layout()
    path = os.path.join(output_dir, "attention_weights.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_prediction_confidence(y_true: np.ndarray, y_prob: np.ndarray, output_dir: str):
    """Histogram of predicted probabilities split by true class.
    Shows whether the model is confident and well-calibrated."""
    fig, ax = plt.subplots(figsize=(8, 5))

    legit_probs = y_prob[y_true == 0]
    phish_probs = y_prob[y_true == 1]

    ax.hist(legit_probs, bins=30, alpha=0.7, color="#4CAF50", label="Legitimate", edgecolor="black")
    ax.hist(phish_probs, bins=30, alpha=0.7, color="#F44336", label="Phishing", edgecolor="black")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Default threshold (0.5)")
    ax.set_xlabel("Predicted Phishing Probability")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Confidence Distribution by True Class")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "prediction_confidence.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_feature_correlation(features: dict, df: pd.DataFrame, output_dir: str):
    """Correlation heatmap of hand-crafted URL + HTML features with the label."""
    url_feat_names = ["url_length", "dots", "hyphens", "slashes", "digits",
                      "special_chars", "has_ip", "has_https", "subdomain_count"]
    html_feat_names = ["form_count", "input_count", "has_password", "script_count",
                       "iframe_count", "meta_refresh", "ext_link_ratio", "text_length"]

    url_np = features["url_features"].numpy()
    html_np = features["html_features"].numpy()
    labels = features["labels"].numpy()

    all_features = np.concatenate([url_np, html_np, labels.reshape(-1, 1)], axis=1)
    col_names = url_feat_names + html_feat_names + ["label"]
    feat_df = pd.DataFrame(all_features, columns=col_names)

    corr = feat_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, ax=ax, vmin=-1, vmax=1,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap (URL + HTML Features + Label)")
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_correlation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_results_summary(all_results: Dict[str, Dict], output_dir: str):
    """Save a clean CSV + text summary of all model metrics."""
    rows = []
    for name, data in all_results.items():
        m = data["metrics"]
        rows.append({
            "Model": name,
            "F1": round(m["f1"], 4),
            "ROC-AUC": round(m["roc_auc"], 4),
            "Accuracy": round(m["accuracy"], 4),
            "Precision": round(m["precision"], 4),
            "Recall": round(m["recall"], 4),
            "PR-AUC": round(m.get("pr_auc", 0), 4),
            "Threshold": round(m["threshold"], 2),
            "TP": m["tp"],
            "FP": m["fp"],
            "FN": m["fn"],
            "TN": m["tn"],
        })

    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "results_summary.csv")
    results_df.to_csv(csv_path, index=False)

    # Also save a formatted text version
    txt_path = os.path.join(output_dir, "results_summary.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PHISHING DETECTION - FINAL RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        for _, row in results_df.iterrows():
            f.write(f"Model: {row['Model']}\n")
            f.write(f"  F1: {row['F1']:.4f}  |  ROC-AUC: {row['ROC-AUC']:.4f}  |  Accuracy: {row['Accuracy']:.4f}\n")
            f.write(f"  Precision: {row['Precision']:.4f}  |  Recall: {row['Recall']:.4f}  |  PR-AUC: {row['PR-AUC']:.4f}\n")
            f.write(f"  Threshold: {row['Threshold']:.2f}\n")
            f.write(f"  Confusion: TP={row['TP']}  FP={row['FP']}  FN={row['FN']}  TN={row['TN']}\n\n")

    return csv_path, txt_path


def save_misclassification_analysis(
    all_results: Dict[str, Dict], df: pd.DataFrame, output_dir: str
):
    """Save examples of misclassified samples for error analysis."""
    if "Multimodal" not in all_results:
        return None

    data = all_results["Multimodal"]
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    y_prob = data["y_prob"]

    test_df = df[df["split"] == "test"].reset_index(drop=True)
    if len(test_df) != len(y_true):
        return None

    test_df = test_df.copy()
    test_df["predicted"] = y_pred
    test_df["prob_phishing"] = np.round(y_prob, 4)
    test_df["correct"] = (y_true == y_pred)

    # False positives: legitimate flagged as phishing
    fp = test_df[(test_df["label"] == 0) & (test_df["predicted"] == 1)].copy()
    fp = fp.sort_values("prob_phishing", ascending=False)

    # False negatives: phishing missed
    fn = test_df[(test_df["label"] == 1) & (test_df["predicted"] == 0)].copy()
    fn = fn.sort_values("prob_phishing", ascending=True)

    txt_path = os.path.join(output_dir, "misclassification_analysis.txt")
    with open(txt_path, "w") as f:
        f.write("MISCLASSIFICATION ANALYSIS (Multimodal Model)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total test samples: {len(test_df)}\n")
        f.write(f"Correct: {test_df['correct'].sum()} ({test_df['correct'].mean()*100:.1f}%)\n")
        f.write(f"False Positives (legit flagged as phishing): {len(fp)}\n")
        f.write(f"False Negatives (phishing missed): {len(fn)}\n\n")

        f.write("-" * 70 + "\n")
        f.write(f"TOP FALSE POSITIVES (most confident mistakes)\n")
        f.write("-" * 70 + "\n")
        for _, row in fp.head(10).iterrows():
            url = str(row["url"])[:100]
            f.write(f"  URL: {url}\n")
            f.write(f"  P(phishing): {row['prob_phishing']:.4f}\n\n")

        f.write("-" * 70 + "\n")
        f.write(f"TOP FALSE NEGATIVES (phishing the model missed)\n")
        f.write("-" * 70 + "\n")
        for _, row in fn.head(10).iterrows():
            url = str(row["url"])[:100]
            f.write(f"  URL: {url}\n")
            f.write(f"  P(phishing): {row['prob_phishing']:.4f}\n\n")

    return txt_path


def save_model_architecture_summary(model, config: dict, output_dir: str):
    """Save model architecture details and parameter counts."""
    txt_path = os.path.join(output_dir, "model_architecture.txt")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with open(txt_path, "w") as f:
        f.write("MODEL ARCHITECTURE SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n\n")

        f.write("Configuration:\n")
        f.write(f"  Fusion strategy: {config['fusion']['strategy']}\n")
        f.write(f"  Projected dimension: {config['fusion']['projected_dim']}\n")
        f.write(f"  Hidden dimension: {config['fusion']['hidden_dim']}\n")
        f.write(f"  Dropout: {config['fusion']['dropout']}\n")
        f.write(f"  Learning rate: {config['training']['learning_rate']}\n")
        f.write(f"  Batch size: {config['training']['batch_size']}\n")
        f.write(f"  Weight decay: {config['training']['weight_decay']}\n\n")

        f.write("Modalities:\n")
        f.write(f"  URL: BiLSTM ({config['url']['hidden_dim']}d x2 bidir) + 9 hand-crafted features\n")
        f.write(f"  Text: DistilBERT mean-pooled (768d, frozen)\n")
        f.write(f"  Visual: EfficientNet-B0 (1280d, frozen)\n")
        f.write(f"  HTML: 8 structural features\n\n")

        f.write("Layer-by-layer:\n")
        f.write("-" * 70 + "\n")
        for name, param in model.named_parameters():
            f.write(f"  {name:50s}  {str(list(param.shape)):>20s}  {'trainable' if param.requires_grad else 'frozen'}\n")

    return txt_path


def plot_learning_rate_schedule(config: dict, output_dir: str):
    """Visualize the cosine annealing LR schedule."""
    import torch

    num_epochs = config["training"]["num_epochs"]
    lr = float(config["training"]["learning_rate"])

    # Simulate the schedule
    dummy_param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([dummy_param], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    lrs = []
    for _ in range(num_epochs):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, num_epochs + 1), lrs, color="steelblue", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Cosine Annealing Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "lr_schedule.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path
