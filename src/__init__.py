"""Re-export public API for convenient imports."""

from src.data_loader import load_cifar10, CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD
from src.models import get_model, SimpleCNN, MediumCNN, DeepCNN
from src.train import train_model, train_one_epoch, evaluate
from src.visualize import (
    plot_lr_schedule,
    plot_training_curves,
    plot_augmentation_comparison,
)

__all__ = [
    "load_cifar10",
    "CIFAR10_CLASSES",
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "get_model",
    "SimpleCNN",
    "MediumCNN",
    "DeepCNN",
    "train_model",
    "train_one_epoch",
    "evaluate",
    "plot_lr_schedule",
    "plot_training_curves",
    "plot_augmentation_comparison",
]

