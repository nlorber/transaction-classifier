"""Publication-quality evaluation plots.

Generates: confusion matrix, top-K accuracy curve, per-class precision/recall,
class-frequency histogram, confidence calibration diagram, and SHAP summary.
"""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from ..core.models.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
    }
)


def plot_confusion_matrix(
    y_true: np.ndarray[Any, np.dtype[Any]],
    y_pred: np.ndarray[Any, np.dtype[Any]],
    class_names: list[str],
    output_path: str | Path,
    top_n: int = 20,
) -> None:
    """Render a confusion matrix for the *top_n* most frequent classes."""
    unique, counts = np.unique(y_true, return_counts=True)
    order = np.argsort(-counts)
    top = unique[order[:top_n]]

    mapped_true = np.where(np.isin(y_true, top), y_true, -1)
    mapped_pred = np.where(np.isin(y_pred, top), y_pred, -1)

    labels = list(top) + [-1]
    cm = confusion_matrix(mapped_true, mapped_pred, labels=labels)
    names = [class_names[i] if i >= 0 else "Other" for i in labels]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=matplotlib.colormaps["Blues"])
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(names)),
        yticks=np.arange(len(names)),
        xticklabels=names,
        yticklabels=names,
        ylabel="True",
        xlabel="Predicted",
        title=f"Confusion Matrix (top {top_n} classes)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if top_n <= 20:
        cutoff = cm.max() / 2.0
        for r in range(len(names)):
            for c in range(len(names)):
                ax.text(
                    c,
                    r,
                    format(cm[r, c], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[r, c] > cutoff else "black",
                    fontsize=6,
                )

    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_topk_accuracy(
    y_true: np.ndarray[Any, np.dtype[Any]],
    proba: np.ndarray[Any, np.dtype[Any]],
    output_path: str | Path,
    k_values: list[int] | None = None,
) -> dict[int, float]:
    """Plot a top-K accuracy curve and return the accuracy dict."""
    if k_values is None:
        k_values = [1, 3, 5, 10]

    k_values = [k for k in k_values if k <= proba.shape[1]]
    accs = []
    for k in k_values:
        topk = np.argsort(-proba, axis=1)[:, :k]
        hit = np.any(topk == y_true.reshape(-1, 1), axis=1)
        accs.append(hit.mean())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, accs, "o-", linewidth=2, markersize=8)
    for k, a in zip(k_values, accs, strict=True):
        ax.annotate(
            f"{a:.1%}", (k, a), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9
        )
    ax.set(xlabel="K", ylabel="Accuracy", title="Top-K Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved %s", output_path)
    return dict(zip(k_values, accs, strict=True))


def plot_per_class_performance(
    y_true: np.ndarray[Any, np.dtype[Any]],
    y_pred: np.ndarray[Any, np.dtype[Any]],
    class_names: list[str],
    output_path: str | Path,
    top_n: int = 20,
) -> None:
    """Precision / recall bar chart for the top-N most frequent classes."""
    unique, counts = np.unique(y_true, return_counts=True)
    top = unique[np.argsort(-counts)[:top_n]]

    prec, rec, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=top,
        zero_division=0,
    )
    names = [class_names[c] for c in top]
    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w / 2, prec, w, label="Precision", color="#2196F3")
    ax.bar(x + w / 2, rec, w, label="Recall", color="#FF9800")
    ax.set(
        xlabel="Account Code", ylabel="Score", title=f"Per-Class Precision & Recall (top {top_n})"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_class_distribution(
    y_true: np.ndarray[Any, np.dtype[Any]],
    output_path: str | Path,
) -> None:
    """Log-scale class frequency histogram."""
    _, counts = np.unique(y_true, return_counts=True)
    ordered = np.sort(counts)[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(ordered)), ordered, color="#4CAF50", alpha=0.8)
    ax.set_yscale("log")
    ax.set(
        xlabel="Class rank",
        ylabel="Samples (log)",
        title=f"Class Distribution ({len(ordered)} classes)",
    )
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_confidence_calibration(
    y_true: np.ndarray[Any, np.dtype[Any]],
    proba: np.ndarray[Any, np.dtype[Any]],
    output_path: str | Path,
    n_bins: int = 10,
) -> None:
    """Reliability diagram for model confidence."""
    y_hat = np.argmax(proba, axis=1)
    conf = np.max(proba, axis=1)
    correct = (y_hat == y_true).astype(float)

    edges = np.linspace(0, 1, n_bins + 1)
    bin_acc, bin_conf, bin_n = [], [], []

    for i in range(n_bins):
        mask = (conf >= edges[i]) & (conf < edges[i + 1])
        if mask.sum() > 0:
            bin_acc.append(correct[mask].mean())
            bin_conf.append(conf[mask].mean())
            bin_n.append(mask.sum())
        else:
            bin_acc.append(0)
            bin_conf.append((edges[i] + edges[i + 1]) / 2)
            bin_n.append(0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax1.plot(bin_conf, bin_acc, "o-", linewidth=2, markersize=6, label="Model")
    ax1.set(xlabel="Mean confidence", ylabel="Fraction correct", title="Calibration")
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    ax2.bar(bin_conf, bin_n, width=0.8 / n_bins, color="#9C27B0", alpha=0.7)
    ax2.set(xlabel="Mean confidence", ylabel="Count")
    ax2.grid(True, alpha=0.3, axis="y")
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_shap_summary(
    model: XGBoostModel,
    X_dense: np.ndarray[Any, np.dtype[Any]],  # noqa: N803
    feature_names: list[str],
    output_path: str | Path,
    max_display: int = 20,
) -> None:
    """SHAP importance bar chart for the top features (averaged across classes)."""
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping. Install with: uv sync --extra explain")
        return

    if model.model is None:
        raise ValueError("Model has not been fitted — cannot compute SHAP values")
    explainer = shap.TreeExplainer(model.model.get_booster())
    sv = explainer.shap_values(X_dense)

    if isinstance(sv, list):
        mean_abs = np.mean([np.abs(a) for a in sv], axis=0)
    else:
        abs_sv = np.abs(sv)
        mean_abs = np.mean(abs_sv, axis=2) if abs_sv.ndim == 3 else abs_sv

    n_feat = mean_abs.shape[1] if mean_abs.ndim == 2 else mean_abs.shape[0]
    if len(feature_names) != n_feat:
        logger.warning(
            "feature_names length (%d) != SHAP count (%d)",
            len(feature_names),
            n_feat,
        )

    importance = mean_abs.mean(axis=0)
    top_idx = np.argsort(importance)[-max_display:][::-1]
    top_names = [feature_names[i] if i < len(feature_names) else f"f_{i}" for i in top_idx]
    top_vals = importance[top_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_names)), top_vals[::-1], color="#E91E63", alpha=0.8)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set(xlabel="Mean |SHAP|", title=f"Top {len(top_names)} Features")
    ax.grid(True, alpha=0.3, axis="x")
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved %s", output_path)


def generate_all_reports(
    y_true: np.ndarray[Any, np.dtype[Any]],
    y_pred: np.ndarray[Any, np.dtype[Any]],
    proba: np.ndarray[Any, np.dtype[Any]],
    class_names: list[str],
    output_dir: str | Path,
    model: XGBoostModel | None = None,
    X_dense: np.ndarray[Any, np.dtype[Any]] | None = None,  # noqa: N803
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Produce the full evaluation report suite and save to *output_dir*."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(y_true, y_pred, class_names, out / "confusion_matrix.png")
    topk = plot_topk_accuracy(y_true, proba, out / "topk_accuracy.png")
    plot_per_class_performance(y_true, y_pred, class_names, out / "per_class_performance.png")
    plot_class_distribution(y_true, out / "class_distribution.png")
    plot_confidence_calibration(y_true, proba, out / "confidence_calibration.png")

    if model is not None and X_dense is not None and feature_names is not None:
        plot_shap_summary(model, X_dense, feature_names, out / "shap_summary.png")

    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "n_classes": len(class_names),
        "n_samples": len(y_true),
        "top_k_accuracy": {str(k): float(v) for k, v in topk.items()},
    }

    metrics_file = out / "metrics.json"
    metrics_file.write_text(json.dumps(summary, indent=2))
    logger.info("Saved %s", metrics_file)

    return summary
