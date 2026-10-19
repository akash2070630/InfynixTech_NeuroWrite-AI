"""Matplotlib/Seaborn figures shared by the training pipeline and the app."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # headless backend: works in scripts, tests and Streamlit

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

from utils.config import (  # noqa: E402
    BAD, BLUE, CLASS_NAMES, GOOD, GRID, NAVY, NUM_CLASSES, PALE, SKY, SLATE,
)


def apply_style() -> None:
    sns.set_theme(style="whitegrid", rc={
        "axes.edgecolor": GRID, "grid.color": GRID, "axes.labelcolor": NAVY,
        "text.color": NAVY, "xtick.color": SLATE, "ytick.color": SLATE,
        "axes.titleweight": "bold", "axes.titlesize": 12, "font.size": 10,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def save_figure(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Dataset figures
# --------------------------------------------------------------------------- #
def plot_sample_grid(x: np.ndarray, y: np.ndarray, per_class: int = 8, seed: int = 0):
    """Grid of real images: one row per digit class."""
    apply_style()
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(NUM_CLASSES, per_class, figsize=(per_class * 0.95, NUM_CLASSES * 0.95))
    for digit in range(NUM_CLASSES):
        picks = rng.choice(np.flatnonzero(y == digit), size=per_class, replace=False)
        for col, idx in enumerate(picks):
            ax = axes[digit, col]
            ax.imshow(x[idx], cmap="gray_r")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col == 0:
                ax.set_ylabel(str(digit), rotation=0, labelpad=12, fontsize=12, color=NAVY, va="center")
    fig.suptitle("MNIST training samples (one row per digit)", color=NAVY, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_class_distribution(train_counts: dict, test_counts: dict):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, counts, title, color in (
        (axes[0], train_counts, "Training set", BLUE),
        (axes[1], test_counts, "Test set", SKY),
    ):
        digits = list(counts.keys())
        values = list(counts.values())
        bars = ax.bar([str(d) for d in digits], values, color=color)
        ax.set_title(f"{title} class counts (n={sum(values):,})")
        ax.set_xlabel("Digit"); ax.set_ylabel("Images")
        ax.set_ylim(0, max(values) * 1.15)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}",
                    ha="center", va="bottom", fontsize=7, color=SLATE)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Training / evaluation figures
# --------------------------------------------------------------------------- #
def plot_training_curves(history: dict):
    apply_style()
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    pairs = (
        (axes[0], "accuracy", "val_accuracy", "Accuracy"),
        (axes[1], "loss", "val_loss", "Loss"),
    )
    for ax, train_key, val_key, title in pairs:
        ax.plot(epochs, history[train_key], color=BLUE, marker="o", ms=3, label="Training")
        ax.plot(epochs, history[val_key], color="#F59E0B", marker="o", ms=3, label="Validation")
        ax.set_title(f"{title} per epoch")
        ax.set_xlabel("Epoch"); ax.set_ylabel(title)
        ax.set_xticks(epochs if len(epochs) <= 20 else ax.get_xticks())
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray):
    """Counts (left) and row-normalised recall (right)."""
    apply_style()
    normalised = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, linewidths=0.4, linecolor="white")
    axes[0].set_title("Confusion matrix (counts)")
    sns.heatmap(normalised, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=axes[1],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, vmin=0, vmax=1,
                linewidths=0.4, linecolor="white")
    axes[1].set_title("Confusion matrix (row-normalised = per-class recall)")
    for ax in axes:
        ax.set_xlabel("Predicted digit"); ax.set_ylabel("True digit")
    fig.tight_layout()
    return fig


def plot_per_class_metrics(report: dict):
    apply_style()
    classes = CLASS_NAMES
    width = 0.27
    pos = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for offset, key, color in ((-width, "precision", NAVY), (0, "recall", BLUE), (width, "f1-score", SKY)):
        ax.bar(pos + offset, [report[c][key] for c in classes], width, label=key.title(), color=color)
    lowest = min(report[c][k] for c in classes for k in ("precision", "recall", "f1-score"))
    ax.set_ylim(max(0.0, min(0.95, lowest - 0.01)), 1.0005)
    ax.set_xticks(pos); ax.set_xticklabels(classes)
    ax.set_xlabel("Digit"); ax.set_ylabel("Score")
    ax.set_title("Per-class precision, recall and F1 (test set; y-axis zoomed)")
    ax.legend(frameon=False, ncol=3, loc="lower center")
    fig.tight_layout()
    return fig


def plot_prediction_examples(x, y_true, y_pred, confidence, title: str, max_items: int = 12, ncols: int = 6):
    """Grid of images captioned with predicted / true label and confidence."""
    apply_style()
    n = min(len(x), max_items)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 1.6))
        ax.axis("off")
        ax.text(0.5, 0.5, "No examples to display", ha="center", va="center", color=SLATE)
        return fig
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.9, nrows * 2.3), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i in range(n):
        ax = axes.ravel()[i]
        correct = y_true[i] == y_pred[i]
        ax.imshow(x[i], cmap="gray_r")
        ax.set_title(f"pred {y_pred[i]} | true {y_true[i]}\n{confidence[i] * 100:.1f}%",
                     fontsize=9, color=GOOD if correct else BAD)
    fig.suptitle(title, color=NAVY, fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# App figure
# --------------------------------------------------------------------------- #
def plot_probabilities(probs: np.ndarray, predicted: int):
    """Horizontal bar chart of the 10 softmax probabilities."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    colors = [BLUE if i == predicted else PALE for i in range(NUM_CLASSES)]
    bars = ax.barh(CLASS_NAMES[::-1], probs[::-1] * 100, color=colors[::-1], edgecolor=SKY, linewidth=0.5)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Probability (%)"); ax.set_ylabel("Digit")
    ax.set_title("Class probabilities (softmax output)")
    for bar, value in zip(bars, probs[::-1] * 100):
        ax.text(min(value + 1.5, 98), bar.get_y() + bar.get_height() / 2, f"{value:.1f}%",
                va="center", fontsize=8, color=SLATE)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig
