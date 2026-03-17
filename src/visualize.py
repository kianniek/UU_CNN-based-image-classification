"""
Visualisation helpers
=====================
Plotting functions for training curves, LR schedules, and augmentation comparisons.
All plots are saved to ``results/`` by default.
"""

import json
import os
from typing import Dict, List, Optional
import torch
import seaborn as sns
import matplotlib.pyplot as plt


RESULTS_DIR = "results"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ------------------------------------------------------------------
# Choice 1 — LR Decay vs. Epochs
# ------------------------------------------------------------------
def plot_lr_schedule(
    lr_history: List[float],
    save_path: Optional[str] = None,
    title: str = "Learning Rate Decay vs. Epochs",
) -> None:
    """Plot the learning-rate schedule across training epochs.

    Parameters
    ----------
    lr_history : list[float]
        One LR value per epoch (recorded at the *start* of each epoch).
    save_path : str or None
        File path for the saved figure. Defaults to ``results/lr_schedule.png``.
    """
    if save_path is None:
        _ensure_dir(RESULTS_DIR)
        save_path = os.path.join(RESULTS_DIR, "lr_schedule.png")

    epochs = list(range(1, len(lr_history) + 1))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, lr_history, marker="o", linewidth=2, color="#1f77b4")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"LR schedule plot saved → {save_path}")


# ------------------------------------------------------------------
# Training curves (loss & accuracy)
# ------------------------------------------------------------------
def plot_training_curves(
    history: Dict[str, List[float]],
    model_name: str = "model",
    save_dir: Optional[str] = None,
) -> None:
    """Plot train/val loss and accuracy curves side by side.

    Parameters
    ----------
    history : dict
        Must contain keys ``train_loss``, ``val_loss``, ``train_acc``, ``val_acc``.
    model_name : str
        Used in the plot title and filename.
    save_dir : str or None
        Directory for the figure. Defaults to ``results/``.
    """
    if save_dir is None:
        save_dir = RESULTS_DIR
    _ensure_dir(save_dir)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"], label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{model_name} — Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = os.path.join(save_dir, f"{model_name}_curves.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Training curves saved → {fname}")


# ------------------------------------------------------------------
# Choice 5 — Augmentation comparison
# ------------------------------------------------------------------
def plot_augmentation_comparison(
    history_aug: Dict[str, List[float]],
    history_no_aug: Dict[str, List[float]],
    model_name: str = "model",
    save_dir: Optional[str] = None,
) -> None:
    """Compare training with vs. without data augmentation.

    Produces a 1×2 figure: validation loss and validation accuracy,
    each with two curves (augmented vs. no augmentation).

    Parameters
    ----------
    history_aug : dict
        Training history *with* augmentation.
    history_no_aug : dict
        Training history *without* augmentation.
    model_name : str
        Used in the plot title and filename.
    save_dir : str or None
        Directory for the figure. Defaults to ``results/``.
    """
    if save_dir is None:
        save_dir = RESULTS_DIR
    _ensure_dir(save_dir)

    epochs_aug = list(range(1, len(history_aug["val_loss"]) + 1))
    epochs_no = list(range(1, len(history_no_aug["val_loss"]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Val loss
    ax1.plot(epochs_aug, history_aug["val_loss"], label="With Augmentation")
    ax1.plot(epochs_no, history_no_aug["val_loss"], label="Without Augmentation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Loss")
    ax1.set_title(f"{model_name} — Val Loss: Augmented vs. Not")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Val accuracy
    ax2.plot(epochs_aug, history_aug["val_acc"], label="With Augmentation")
    ax2.plot(epochs_no, history_no_aug["val_acc"], label="Without Augmentation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy (%)")
    ax2.set_title(f"{model_name} — Val Acc: Augmented vs. Not")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = os.path.join(save_dir, f"{model_name}_augmentation_comparison.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Augmentation comparison saved → {fname}")

def plot_multi_model_comparison(
    metadata_paths: List[str],
    metric: str = "val_acc",
    save_path: Optional[str] = None,
) -> None:
    """
    Compares multiple models by plotting a specific metric from their metadata files.
    
    Parameters
    ----------
    metadata_paths : list[str]
        List of paths to JSON metadata files.
    metric : str
        The key in the history dict to plot (e.g., 'val_acc', 'val_loss').
    """
    if save_path is None:
        _ensure_dir(RESULTS_DIR)
        save_path = os.path.join(RESULTS_DIR, f"comparison_{metric}.png")

    fig, ax = plt.subplots(figsize=(10, 6))

    for path in metadata_paths:
        with open(path, "r") as f:
            data = json.load(f)
        
        history = data["history"]
        label = data.get("label", os.path.basename(path))
        epochs = list(range(1, len(history[metric]) + 1))
        
        ax.plot(epochs, history[metric], label=label, linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Comparison: {metric}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Comparison plot saved → {save_path}")


def plot_kfold_results(cv_results, model_name: str = "model", save_dir: Optional[str] = None):
    """Plot k-fold cross-validation results."""
    if save_dir is None:
        save_dir = RESULTS_DIR
    _ensure_dir(save_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    folds = range(1, len(cv_results["accuracies"]) + 1)

    # Accuracy per fold
    ax1.bar(folds, cv_results["accuracies"], color='skyblue', alpha=0.7)
    ax1.axhline(y=cv_results["mean_accuracy"], color='red', linestyle='--', label=f'Mean: {cv_results['mean_accuracy']:.2f}%')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Validation Accuracy (%)')
    ax1.set_title(f'{model_name} - K-Fold Accuracy')
    ax1.set_xticks(folds)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss per fold
    ax2.bar(folds, cv_results["losses"], color='lightcoral', alpha=0.7)
    ax2.axhline(y=cv_results["mean_loss"], color='red', linestyle='--', label=f'Mean: {cv_results['mean_loss']:.4f}')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title(f'{model_name} - K-Fold Loss')
    ax2.set_xticks(folds)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = os.path.join(save_dir, f"{model_name}_kfold_results.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"K-Fold results plot saved → {fname}")


def plot_hyperparameter_search(results, model_name: str = "model", save_dir: Optional[str] = None):
    """Plot hyperparameter search results."""
    if save_dir is None:
        save_dir = RESULTS_DIR
    _ensure_dir(save_dir)

    # Extract data for plotting
    accuracies = [r["val_accuracy"] for r in results["all_results"]]
    labels = [f"{r['optimizer']}\nLR:{r['learning_rate']:.0e}\nBS:{r['batch_size']}" for r in results["all_results"]]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(range(len(accuracies)), accuracies, color='lightblue', edgecolor='navy', alpha=0.7)
    ax.set_xlabel('Hyperparameter Combinations')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title(f'{model_name} - Hyperparameter Search Results')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Highlight best result
    if accuracies:
        best_idx = accuracies.index(max(accuracies))
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('orange')
        bars[best_idx].set_alpha(1.0)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = os.path.join(save_dir, f"{model_name}_hyperparameter_search.png")
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Hyperparameter search plot saved → {fname}")
    
# def plot_confusion_matrix(matrix):
def plot_confusion_matrix(matrix, class_names: Optional[List[str]] = None, model_name: str = "model", save_dir: Optional[str] = None):
    """Plot and save a confusion matrix heatmap.

    Parameters
    - matrix: 2D array-like confusion matrix (true x predicted)
    - class_names: optional list of class labels for axes
    - model_name: used in filename and title
    - save_dir: directory to save the figure (defaults to results/)
    """
    if save_dir is None:
        save_dir = RESULTS_DIR
    _ensure_dir(save_dir)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax)

    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title(f"{model_name} - Confusion Matrix")

    if class_names is not None:
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names, rotation=0)

    fname = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved → {fname}")
    