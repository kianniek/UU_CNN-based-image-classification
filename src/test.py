
import torch
import torch.nn as nn
from sklearn import metrics as sk
from sklearn import manifold as skm
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple

# This is a separate test function called specifically for Choice Task metrics. 
# Model is tested with the test set
@torch.no_grad()
def test(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    """Test the model on *loader*. Returns (avg_loss, accuracy, confusion matrix, TSNE coordinates, labels)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    test_pred = []
    test_label = []
    fully_connected_outputs = []
    logits_outputs = []
    
    # grabs the output from the Fully Connected layer for TSNE coordinates
    def hook_fn(model, input, output):
        fully_connected_outputs.append(output.detach().cpu())
        
    fc_module = getattr(model, "fully_connected", None)
    fc_layer = None
    if isinstance(fc_module, nn.Module):
        fc_layer = fc_module.register_forward_hook(hook_fn)
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        logits_outputs.append(outputs.detach().cpu())
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        prediction = outputs.data.max(1, keepdim=True)[1]
        
        test_pred.append(prediction)
        test_label.append(labels)
    
    if fc_layer is not None:
        fc_layer.remove()
    
    if fully_connected_outputs:
        all_predictions = torch.cat(fully_connected_outputs, dim=0).cpu().numpy()
    else:
        all_predictions = torch.cat(logits_outputs, dim=0).cpu().numpy()
    
    test_pred = torch.cat(test_pred)
    test_label = torch.cat(test_label)
    
    tsne_x, tsne_y =  compute_sne(all_predictions)
    conf_m = compute_confusion_matrix(test_pred, test_label)
    
    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, conf_m, [tsne_x, tsne_y], test_label.cpu().numpy()

def compute_confusion_matrix(predictions, labels):
    """Compute model's confusion matrix with model predictions and labels"""
    conf_m = sk.confusion_matrix(predictions.cpu().numpy(), labels.cpu().numpy())
    
    return conf_m

def compute_sne(images):
    """Compute the probable similarities for each Tensor image for the t-SNE graph"""
    tsne = skm.TSNE(n_components=2).fit_transform(images)
    
    tsne_x = tsne[:,0]
    tsne_y = tsne[:,1]
    
    tsne_x = scale_value(tsne_x)
    tsne_y = scale_value(tsne_y)
    return tsne_x, tsne_y


def scale_value(val):
    """Scales val between a range of 0 and 1"""
    range = (np.max(val) - np.min(val))
    
    zero_start = val - np.min(val)
    
    return zero_start / range