"""
Main entry point for CIFAR-10 CNN training and evaluation.
"""

import argparse
import torch

from src.data_loader import load_cifar10, CIFAR10_CLASSES
from src.models import get_model, MODEL_REGISTRY
from src.train import train_model, evaluate
from src.visualize import (
    plot_lr_schedule,
    plot_training_curves,
    plot_augmentation_comparison,
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
        "--batch-size", type=int, default=32, help="Mini-batch size (default: 32)"
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
    # General
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable training-time data augmentation",
    )
    return parser.parse_args()


def _run_training(args, augment: bool, device: torch.device, tag: str = ""):
    """Helper: load data, build model, train, return history."""
    train_loader, val_loader, test_loader = load_cifar10(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        augment_train=augment,
        seed=args.seed,
    )

    model = get_model(args.model).to(device)
    label = f"{args.model}{tag}"
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
        lr=args.lr,
        use_scheduler=not args.no_scheduler,
        scheduler_step_size=args.scheduler_step,
        scheduler_gamma=args.scheduler_gamma,
        save_path=f"results/{label}_{args.epochs}_{args.lr}.pth",
    )

    # Final test evaluation
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test  Loss {test_loss:.4f}  Acc {test_acc:.2f}%")

    return history, model


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Classes: {CIFAR10_CLASSES}")

    if args.compare_augmentation:
        # ---- Choice 5: train with & without augmentation, then compare ----
        history_aug, _ = _run_training(args, augment=True, device=device, tag="_aug")
        history_no_aug, _ = _run_training(args, augment=False, device=device, tag="_noaug")

        plot_augmentation_comparison(history_aug, history_no_aug, model_name=args.model)
        plot_training_curves(history_aug, model_name=f"{args.model}_aug")
        plot_training_curves(history_no_aug, model_name=f"{args.model}_noaug")

        # Plot LR schedule (same for both runs)
        plot_lr_schedule(history_aug["lr"])
    else:
        # ---- Normal single training run ----
        augment = not args.no_augment
        history, _ = _run_training(args, augment=augment, device=device)

        # Choice 1 — Plot LR decay vs. Epochs
        plot_lr_schedule(history["lr"])
        plot_training_curves(history, model_name=args.model)


if __name__ == "__main__":
    main()
