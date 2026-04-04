import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
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
