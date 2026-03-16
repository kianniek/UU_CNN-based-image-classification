"""
Training Engine
===============
Provides a training loop with:
- Adam optimiser (default LR=0.001, batch_size=32)
- Cross-Entropy loss
- StepLR scheduler (halve LR every 5 epochs)
- Per-epoch train / validation metrics collection
- Early stopping with configurable patience and monitored metric
"""

import copy
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on *loader*. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    use_scheduler: bool = True,
    scheduler_step_size: int = 5,
    scheduler_gamma: float = 0.5,
    save_path: Optional[str] = None,
    early_stopping: bool = False,
    patience: int = 3,
    monitor: str = "val_loss",
    max_epochs: int = 200,
) -> Dict[str, Any]:
    """
    Full training loop with optional early stopping.

    Two modes
    ---------
    **Manual** (default): trains for exactly *epochs* epochs.

    **Automatic** (``early_stopping=True``): trains up to *max_epochs*
    epochs but stops early when the monitored metric has not improved
    for *patience* consecutive epochs.  The *epochs* parameter is
    ignored in this mode.

    Parameters
    ----------
    model : nn.Module
        The network to train (already on *device*).
    train_loader, val_loader : DataLoader
        Data loaders for training and validation.
    device : torch.device
    epochs : int
        Number of epochs (manual mode only).
    lr : float
        Initial learning rate for Adam.
    weight_decay : float
        L2 regularisation coefficient.
    use_scheduler : bool
        If True, attach a StepLR scheduler that halves the LR every
        *scheduler_step_size* epochs.
    scheduler_step_size : int
        Epoch interval for LR reduction (default 5).
    scheduler_gamma : float
        Multiplicative factor for LR reduction (default 0.5 → halve).
    save_path : str or None
        If provided, saves the final model state_dict to this path.
    early_stopping : bool
        If True, enable automatic convergence detection.
    patience : int
        Number of epochs without improvement before stopping
        (only used when ``early_stopping=True``).
    monitor : str
        Metric to monitor: ``"val_loss"`` (lower is better) or
        ``"val_acc"`` (higher is better).
    max_epochs : int
        Upper-bound epoch count in automatic mode (default 200).

    Returns
    -------
    history : dict
        Keys: ``train_loss``, ``train_acc``, ``val_loss``, ``val_acc``,
        ``lr``, ``stopped_epoch``.
    """
    if monitor not in ("val_loss", "val_acc"):
        raise ValueError(f"monitor must be 'val_loss' or 'val_acc', got '{monitor}'")
    # Determine total number of epochs
    total_epochs = max_epochs if early_stopping else epochs

    # Softmax is handled by the Cross Entropy loss function. It combines nn.LogSoftmax and nn.NLLLoss
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler: Optional[StepLR] = None
    if use_scheduler:
        scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    history: Dict[str, Any] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }
    
    

    # Best-model tracking (metric-aware)
    monitor_lower_is_better = monitor == "val_loss"
    best_metric = float("inf") if monitor_lower_is_better else 0.0
    best_weights = None
    best_epoch = 0
    epochs_without_improvement = 0
    converged = False

    if early_stopping:
        print(
            f"Early stopping enabled: monitor={monitor}, "
            f"patience={patience}, max_epochs={max_epochs}"
        )

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()

        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:>3d}/{total_epochs} | "
            f"LR {current_lr:.6f} | "
            f"Train Loss {train_loss:.4f}  Acc {train_acc:6.2f}% | "
            f"Val Loss {val_loss:.4f}  Acc {val_acc:6.2f}% | "
            f"{elapsed:.1f}s"
        )

        # Track best model based on monitored metric
        current_metric = val_loss if monitor_lower_is_better else val_acc
        improved = (
            current_metric < best_metric
            if monitor_lower_is_better
            else current_metric > best_metric
        )

        if improved:
            best_metric = current_metric
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping check
        if early_stopping and epochs_without_improvement >= patience:
            print(
                f"\nConverged: {monitor} has not improved for "
                f"{patience} epochs (best {monitor}="
                f"{best_metric:.4f} at epoch {best_epoch})"
            )
            converged = True
            break

        if scheduler is not None:
            scheduler.step()

    # Record which epoch training actually stopped at
    stopped_epoch = epoch
    history["stopped_epoch"] = stopped_epoch

    # Summary
    if early_stopping and not converged:
        print(
            f"\nReached max epochs ({max_epochs}) without convergence. "
            f"Best {monitor}={best_metric:.4f} at epoch {best_epoch}."
        )

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
        metric_label = "val loss" if monitor_lower_is_better else "val acc"
        unit = "" if monitor_lower_is_better else "%"
        print(f"Restored best model (epoch {best_epoch}, {metric_label} {best_metric:.4f}{unit})")

    # Save final model state_dict if requested
    if save_path is not None:
        import os

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved → {save_path}")

    return history

