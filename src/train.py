"""
Training Engine
===============
Provides a training loop with:
- Adam optimiser (default LR=0.001, batch_size=32)
- Cross-Entropy loss
- StepLR scheduler (halve LR every 5 epochs)
- Per-epoch train / validation metrics collection
"""

import copy
import time
from typing import Dict, List, Optional, Tuple

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
) -> Dict[str, List[float]]:
    """
    Full training loop.

    Parameters
    ----------
    model : nn.Module
        The network to train (already on *device*).
    train_loader, val_loader : DataLoader
        Data loaders for training and validation.
    device : torch.device
    epochs : int
        Number of epochs.
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

    Returns
    -------
    history : dict
        Keys: ``train_loss``, ``train_acc``, ``val_loss``, ``val_acc``,
        ``lr`` (learning rate per epoch).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler: Optional[StepLR] = None
    if use_scheduler:
        scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
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
            f"Epoch {epoch:>3d}/{epochs} | "
            f"LR {current_lr:.6f} | "
            f"Train Loss {train_loss:.4f}  Acc {train_acc:6.2f}% | "
            f"Val Loss {val_loss:.4f}  Acc {val_acc:6.2f}% | "
            f"{elapsed:.1f}s"
        )

        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

        if scheduler is not None:
            scheduler.step()

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"\nRestored best model (val acc {best_val_acc:.2f}%)")

    # Save final model state_dict if requested
    if save_path is not None:
        import os

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved → {save_path}")

    return history
