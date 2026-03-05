"""
Evaluation utilities: top-1 accuracy, confusion matrix computation.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, List[int], List[int]]:
    """Run inference on *loader* and return accuracy + prediction lists.

    Args:
        model:   Trained neural network.
        loader:  DataLoader over the evaluation split.
        device:  torch.device.

    Returns:
        Tuple of (accuracy, all_targets, all_predictions).
        ``accuracy`` is the fraction of correctly classified samples.
    """
    model.eval()
    all_targets: List[int] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_targets.extend(labels.cpu().tolist())
            all_preds.extend(predicted.cpu().tolist())

    correct = sum(t == p for t, p in zip(all_targets, all_preds))
    accuracy = correct / len(all_targets)
    return accuracy, all_targets, all_preds


def compute_confusion_matrix(
    targets: List[int],
    predictions: List[int],
    num_classes: int,
) -> torch.Tensor:
    """Return a *num_classes* × *num_classes* confusion matrix.

    Entry [i, j] is the number of samples whose true class is *i* and
    predicted class is *j*.
    """
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(targets, predictions):
        matrix[t][p] += 1
    return matrix
