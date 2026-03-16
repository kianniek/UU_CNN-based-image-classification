import copy
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import sklearn as sk
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


@torch.no_grad()
def test(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Test the model on *loader*. Returns (avg_loss, accuracy, predictions, labels)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    test_pred = []
    test_label = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        prediction = outputs.data.max(1, keepdim=True)[1]
        test_pred.append(prediction)
        test_label.append(labels)
        
    conf_m = compute_confusion_matrix(test_pred, test_label)
    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, conf_m

def compute_confusion_matrix(predictions, labels):
    predictions = torch.cat(predictions)
    labels = torch.cat(labels)
    
    conf_m = sk.metrics.confusion_matrix(predictions.cpu().numpy(), labels.cpu().numpy())
    
    return conf_m