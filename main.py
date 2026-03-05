"""
Main entry point for CIFAR-10 CNN training and evaluation.
"""

import argparse
import torch

from src.data_loader import load_cifar10, CIFAR10_CLASSES
from src.models import get_model, MODEL_REGISTRY


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
        "--batch-size", type=int, default=64, help="Mini-batch size (default: 64)"
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
    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = load_cifar10(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model = get_model(args.model).to(device)
    print(f"\nModel: {args.model}")
    print(model)

    print(f"\nClasses: {CIFAR10_CLASSES}")
    images, labels = next(iter(train_loader))
    print(f"Sample batch — images: {images.shape}, labels: {labels.shape}")


if __name__ == "__main__":
    main()
