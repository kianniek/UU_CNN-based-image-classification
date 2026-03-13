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
from torch.optim import Adam, SGD, RMSprop
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


def k_fold_cross_validation(
    model_class,
    full_dataset,
    device: torch.device,
    k: int = 5,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation on the given dataset.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    fold_results = []
    fold_accuracies = []
    fold_losses = []

    print(f"Starting {k}-fold cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        print(f"\n=== Fold {fold + 1}/{k} ===")

        # Create train/val splits for this fold
        train_subset = torch.utils.data.Subset(full_dataset, train_idx.tolist())
        val_subset = torch.utils.data.Subset(full_dataset, val_idx.tolist())

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Create new model instance for this fold
        model = model_class()
        model.to(device)

        # Train model
        history = train_model(
            model, train_loader, val_loader, device, epochs=epochs, lr=lr, weight_decay=weight_decay, **kwargs
        )

        # Store results
        final_val_acc = history["val_acc"][-1]
        final_val_loss = history["val_loss"][-1]

        fold_results.append({
            "fold": fold + 1,
            "val_accuracy": final_val_acc,
            "val_loss": final_val_loss,
            "history": history,
        })

        fold_accuracies.append(final_val_acc)
        fold_losses.append(final_val_loss)

        print(f"Fold {fold + 1} - Val Accuracy: {final_val_acc:.2f}%, Val Loss: {final_val_loss:.4f}")

    # Calculate statistics
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_loss = np.mean(fold_losses)
    std_loss = np.std(fold_losses)

    results = {
        "fold_results": fold_results,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_loss": mean_loss,
        "std_loss": std_loss,
        "accuracies": fold_accuracies,
        "losses": fold_losses,
    }

    print(f"\n{k}-Fold CV Results:")
    print(f"Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Mean Loss: {mean_loss:.4f} ± {std_loss:.4f}")

    return results


def hyperparameter_search(
    model_class,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    optimizers: List[str] = ["adam", "sgd", "rmsprop"],
    learning_rates: List[float] = [1e-3, 1e-4, 1e-5],
    weight_decays: List[float] = [0.0, 1e-4],
    batch_sizes: List[int] = [16, 32],
    epochs: int = 20,
    max_search_points: int = 5,
) -> Dict[str, Any]:
    """
    Perform grid search over hyperparameters.
    """
    from itertools import product

    # Generate all combinations (limited for grid search)
    param_combinations = list(product(optimizers, learning_rates, weight_decays, batch_sizes))

    # Limit combinations for grid search
    if len(param_combinations) > max_search_points:
        indices = np.linspace(0, len(param_combinations) - 1, max_search_points, dtype=int)
        param_combinations = [param_combinations[i] for i in indices]

    results = []

    print(f"Starting hyperparameter search with {len(param_combinations)} combinations...")

    for i, (opt_name, lr, wd, bs) in enumerate(param_combinations):
        print(f"\n--- Combination {i+1}/{len(param_combinations)} ---")
        print(f"Optimizer: {opt_name}, LR: {lr}, Weight Decay: {wd}, Batch Size: {bs}")

        # Update data loaders if batch size changed
        if bs != train_loader.batch_size:
            train_loader_new = DataLoader(
                train_loader.dataset,
                batch_size=bs,
                shuffle=True,
                num_workers=getattr(train_loader, "num_workers", 0),
                pin_memory=True,
            )
            val_loader_new = DataLoader(
                val_loader.dataset,
                batch_size=bs,
                shuffle=False,
                num_workers=getattr(val_loader, "num_workers", 0),
                pin_memory=True,
            )
        else:
            train_loader_new = train_loader
            val_loader_new = val_loader

        # Create model
        model = model_class()
        model.to(device)

        # Setup optimizer
        if opt_name.lower() == "adam":
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name.lower() == "sgd":
            optimizer = SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        elif opt_name.lower() == "rmsprop":
            optimizer = RMSprop(model.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        # Setup loss and scheduler
        criterion = nn.CrossEntropyLoss()
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

        # Train model
        best_val_acc = 0.0
        best_weights = None

        for epoch in range(epochs):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader_new, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader_new, criterion, device)

            elapsed = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Combination {i+1}/{len(param_combinations)} | "
                f"Epoch {epoch + 1:>3d}/{epochs} | "
                f"LR {current_lr:.6f} | "
                f"Train Loss {train_loss:.4f}  Acc {train_acc:6.2f}% | "
                f"Val Loss {val_loss:.4f}  Acc {val_acc:6.2f}% | "
                f"{elapsed:.1f}s"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = copy.deepcopy(model.state_dict())

            scheduler.step()

        # Store results
        results.append({
            "optimizer": opt_name,
            "learning_rate": lr,
            "weight_decay": wd,
            "batch_size": bs,
            "val_accuracy": best_val_acc,
            "val_loss": val_loss,
        })

        print(f"Result: Val Accuracy = {best_val_acc:.2f}%, Val Loss = {val_loss:.4f}")

    # Sort by validation accuracy (descending)
    results.sort(key=lambda x: x["val_accuracy"], reverse=True)

    print("\nHyperparameter Search Results (sorted by validation accuracy):")
    print("-" * 80)
    print(f"{'Rank':<4} {'Optimizer':<10} {'LR':<10} {'WD':<10} {'Batch Size':<10} {'Val Acc':<10} {'Val Loss':<10}")
    print("-" * 80)
    for i, result in enumerate(results, 1):
        print(f"{i:<4} {result['optimizer']:<10} {result['learning_rate']:<10.1e} "
            f"{result['weight_decay']:<10.1e} {result['batch_size']:<10} "
            f"{result['val_accuracy']:<10.2f} {result['val_loss']:<10.4f}")

    return {"all_results": results, "best_combination": results[0] if results else None}

