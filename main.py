"""
Main entry point for CIFAR-10 CNN training and evaluation.
"""

import argparse
import torch
import json
import os
from datetime import datetime

from src.data_loader import load_cifar10, CIFAR10_CLASSES, load_cifar100, CIFAR100_SUPERCLASSES
from src.models import get_model, MODEL_REGISTRY
from src.train import train_model, evaluate
from src.visualize import (
    plot_lr_schedule,
    plot_training_curves,
    plot_augmentation_comparison,
    plot_multi_model_comparison,  # New import
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate CNN models on CIFAR-10"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="simple",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model architecture to use (default: simple)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Mini-batch size (default: 32) "
        "Standard batch sizes in literature have been: 32, 64, 128, 256"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="Dataset cache directory"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    
    # Choice 1 — LR Scheduler
    parser.add_argument(
        "--no-scheduler",
        action="store_true",
        help="Disable the StepLR scheduler (enabled by default: halve LR every 5 epochs)",
    )
    parser.add_argument(
        "--scheduler-step", type=int, default=5,
        help="StepLR step size in epochs (default: 5)",
    )
    parser.add_argument(
        "--scheduler-gamma", type=float, default=0.5,
        help="StepLR multiplicative factor (default: 0.5 = halve)",
    )
    # Choice 5 — Data Augmentation comparison
    parser.add_argument(
        "--compare-augmentation",
        action="store_true",
        help="Train twice (with & without augmentation) and plot the comparison",
    )
    # Metadata comparison
    parser.add_argument(
        "--compare-models",
        nargs="+",
        help="Paths to metadata JSON files to compare (e.g. results/model1_metadata.json results/model2_metadata.json)",
    )
    # Early stopping / convergence detection
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable automatic convergence detection (early stopping)",
    )
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Epochs without improvement before stopping (default: 3)",
    )
    parser.add_argument(
        "--monitor", type=str, default="val_loss",
        choices=["val_loss", "val_acc"],
        help="Metric to monitor for early stopping (default: val_loss)",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=200,
        help="Max epochs safety cap in automatic mode (default: 200)",
    )
    # General
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable training-time data augmentation",
    )
    return parser.parse_args()

def save_metadata(args, history, label, save_dir="results"):
    """Saves hyperparameters and training history to a JSON file."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Combine args and history into one serializable dict
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "history": history,
        "label": label
    }
    
    file_path = os.path.join(save_dir, f"{label}_metadata.json")
    with open(file_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Metadata saved → {file_path}")
    return file_path


def _augmentation_suffix(augment: bool) -> str:
    """Return a stable filename suffix that reflects augmentation settings."""
    return "augmented" if augment else "not-augmented"


def _load_pretrained_backbone_weights(
    model: torch.nn.Module,
    ckpt_path: str,
    device: torch.device,
) -> bool:
    """Load CIFAR-100 weights into matching layers; keep 10-class head random."""
    if not os.path.exists(ckpt_path):
        warning = (
            "WARNING: Pretrained checkpoint not found. "
            "Please provide the required CIFAR-100 checkpoint at the expected path to enable pretrained fine-tuning. "
            "Press Enter to continue without loading weights (will train from scratch)."
        )
        input(warning)
        return False

    state = torch.load(ckpt_path, map_location=device)
    model_state = model.state_dict()

    compatible_state = {
        k: v
        for k, v in state.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    skipped_keys = sorted(set(state.keys()) - set(compatible_state.keys()))

    model_state.update(compatible_state)
    model.load_state_dict(model_state)

    print(
        f"Loaded {len(compatible_state)} layers from CIFAR-100 checkpoint: {ckpt_path}"
    )
    if skipped_keys:
        print("Skipped incompatible layers (expected for final classifier):")
        for key in skipped_keys:
            print(f"  - {key}")
    return True

def _run_training(args, augment: bool, device: torch.device):
    """Helper: load data, build model, train, return history."""
    
    if args.model == "cifar100":
        train_loader, val_loader, test_loader = load_cifar100(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            augment_train=augment,
            seed=args.seed,
        )
    else:
        train_loader, val_loader, test_loader = load_cifar10(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            augment_train=augment,
            seed=args.seed,
        )

    model = get_model(args.model).to(device)

    effective_lr = args.lr
    if args.model == "pretrained":
        source_ckpt = os.path.join("results", "cifar100_20_0.001.pth")
        loaded = _load_pretrained_backbone_weights(model, source_ckpt, device)
        if loaded:
            effective_lr = args.lr * 0.5
            print(
                f"Fine-tuning pretrained model with half LR: {effective_lr:.6f} (base {args.lr:.6f})"
            )
        else:
            print(
                "Pretrained weights were not loaded. "
                f"Continuing with random initialization at base LR: {effective_lr:.6f}."
            )

    label = f"{args.model}_{_augmentation_suffix(augment)}"
    print(f"\n{'='*60}")
    print(f"Training: {label}  |  augmentation={'ON' if augment else 'OFF'}")
    print(f"{'='*60}")
    print(model)

    history = train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=args.epochs,
        lr=effective_lr,
        use_scheduler=not args.no_scheduler,
        scheduler_step_size=args.scheduler_step,
        scheduler_gamma=args.scheduler_gamma,
        save_path=f"results/{label}_{args.epochs}_{effective_lr}.pth",
        early_stopping=args.early_stopping,
        patience=args.patience,
        monitor=args.monitor,
        max_epochs=args.max_epochs,
    )

    # Save metadata for future comparison or graph recreation
    save_metadata(args, history, label)

    # Final test evaluation
    # Softmax is handled by the Cross Entropy loss function
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test  Loss {test_loss:.4f}  Acc {test_acc:.2f}%")

    return history, model

def main():
    args = parse_args()

    # If user wants to compare existing runs, do that and exit
    if args.compare_models:
        plot_multi_model_comparison(args.compare_models, metric="val_acc")
        plot_multi_model_comparison(args.compare_models, metric="val_loss")
        return

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if args.model == "cifar100":
        print(f"Classes: {CIFAR100_SUPERCLASSES}")
    else:
        print(f"Classes: {CIFAR10_CLASSES}")

    if args.compare_augmentation:
        history_aug, _ = _run_training(args, augment=True, device=device)
        history_no_aug, _ = _run_training(args, augment=False, device=device)

        plot_augmentation_comparison(history_aug, history_no_aug, model_name=args.model)
        plot_training_curves(history_aug, model_name=f"{args.model}_{_augmentation_suffix(True)}")
        plot_training_curves(history_no_aug, model_name=f"{args.model}_{_augmentation_suffix(False)}")
        plot_lr_schedule(history_aug["lr"])
    else:
        augment = not args.no_augment
        history, _ = _run_training(args, augment=augment, device=device)
        plot_lr_schedule(history["lr"])
        plot_training_curves(history, model_name=f"{args.model}_{_augmentation_suffix(augment)}")

if __name__ == "__main__":
    main()
