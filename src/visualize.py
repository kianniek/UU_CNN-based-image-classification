"""
Visualisation utilities for training curves, confusion matrices, and
(optionally) t-SNE embeddings.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Training / validation curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: Dict[str, List[float]],
    model_name: str,
    save_path: Optional[str] = None,
):
    """Plot training and validation loss and accuracy over epochs.

    Args:
        history:    Dict returned by :func:`src.train.train_model`.
        model_name: Title prefix for the figure.
        save_path:  If given, save the figure to this path.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} – Loss")
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, history["train_accuracy"], label="Train")
    axes[1].plot(epochs, history["val_accuracy"], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{model_name} – Accuracy")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close(fig)


def plot_lr_curve(
    history: Dict[str, List[float]],
    model_name: str,
    save_path: Optional[str] = None,
):
    """Plot the learning-rate schedule stored in *history* (CHOICE 1).

    Args:
        history:    Dict returned by :func:`src.train.train_model`.
        model_name: Title prefix for the figure.
        save_path:  If given, save the figure to this path.
    """
    if "lr" not in history or not history["lr"]:
        return

    epochs = range(1, len(history["lr"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["lr"], marker="o", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title(f"{model_name} – Learning Rate Schedule")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    confusion_matrix: torch.Tensor,
    class_names: List[str],
    model_name: str,
    save_path: Optional[str] = None,
    normalize: bool = True,
):
    """Plot a confusion matrix as a color-coded grid.

    Args:
        confusion_matrix: Square tensor of shape (C, C).
        class_names:      List of class label strings.
        model_name:       Title prefix.
        save_path:        If given, save the figure to this path.
        normalize:        If True, show row-normalised proportions.
    """
    cm = confusion_matrix.numpy().astype(float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)
        fmt = ".2f"
        vmax = 1.0
    else:
        fmt = "d"
        vmax = cm.max()

    n = len(class_names)
    fig_size = max(6, n * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} – Confusion Matrix")

    thresh = vmax / 2.0
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            text = f"{val:{fmt}}" if fmt == ".2f" else f"{int(val)}"
            color = "white" if val > thresh else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=6, color=color)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table helper
# ---------------------------------------------------------------------------

def print_results_table(results: Dict[str, Dict[str, float]]):
    """Print a formatted table of train/val accuracy for all models.

    Args:
        results: Mapping of model_name → dict with keys
                 ``"train_acc"``, ``"val_acc"``, ``"test_acc"``.
    """
    header = f"{'Model':<25} {'Train Acc':>10} {'Val Acc':>10} {'Test Acc':>10}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, metrics in results.items():
        print(
            f"{name:<25} "
            f"{metrics.get('train_acc', float('nan')):>10.4f} "
            f"{metrics.get('val_acc', float('nan')):>10.4f} "
            f"{metrics.get('test_acc', float('nan')):>10.4f}"
        )
    print(sep)
