"""
Training and validation utilities.

Provides:
    train_one_epoch  – run a single training epoch and return loss / accuracy.
    validate         – evaluate a model on a DataLoader and return loss / accuracy.
    train_model      – full training loop with optional LR scheduling (CHOICE 1).
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Single-epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train *model* for one pass over *loader*.

    Returns:
        dict with keys ``"loss"`` and ``"accuracy"``.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return {"loss": total_loss / total, "accuracy": correct / total}


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate *model* on *loader* without updating weights.

    Returns:
        dict with keys ``"loss"`` and ``"accuracy"``.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)

    return {"loss": total_loss / total, "accuracy": correct / total}


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    use_lr_scheduler: bool = False,
    lr_step_size: int = 5,
    lr_gamma: float = 0.5,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Train *model* and record per-epoch metrics.

    Args:
        model:             The neural network to train (must already be on *device*).
        train_loader:      DataLoader for the training split.
        val_loader:        DataLoader for the validation split.
        num_epochs:        Total number of training epochs.
        learning_rate:     Initial Adam learning rate.
        device:            torch.device to use.
        use_lr_scheduler:  If True, halve the LR every *lr_step_size* epochs
                           (CHOICE 1).
        lr_step_size:      Epochs between LR reductions (default 5).
        lr_gamma:          Multiplicative factor for LR reduction (default 0.5).
        verbose:           Print per-epoch progress.

    Returns:
        History dict with keys:
            ``"train_loss"``, ``"train_accuracy"``,
            ``"val_loss"``,   ``"val_accuracy"``,
            ``"lr"`` (learning rate recorded after each epoch).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler: Optional[StepLR] = None
    if use_lr_scheduler:
        scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "lr": [],
    }

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["lr"].append(current_lr)

        if scheduler is not None:
            scheduler.step()

        if verbose:
            print(
                f"Epoch [{epoch:3d}/{num_epochs}] "
                f"LR: {current_lr:.6f}  "
                f"Train Loss: {train_metrics['loss']:.4f}  "
                f"Train Acc: {train_metrics['accuracy']:.4f}  "
                f"Val Loss: {val_metrics['loss']:.4f}  "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )

    return history
