
import torch
import torch.nn as nn
from sklearn import metrics as sk
from sklearn import manifold as skm
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple


@torch.no_grad()
def test(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    """Test the model on *loader*. Returns (avg_loss, accuracy, predictions, labels)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    test_pred = []
    test_label = []
    fully_connected_outputs = []
    
    def hook_fn(model, input, output):
        fully_connected_outputs.append(output.detach().cpu())
        
    fc_layer = model.fully_connected.register_forward_hook(hook_fn)
    
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
    
    fc_layer.remove()
    
    all_predictions = torch.cat(fully_connected_outputs, dim = 0).cpu().numpy()
    test_pred = torch.cat(test_pred)
    test_label = torch.cat(test_label)
    
    tsne_x, tsne_y =  compute_sne(all_predictions)
    conf_m = compute_confusion_matrix(test_pred, test_label)
    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, conf_m, [tsne_x, tsne_y], test_label.cpu().numpy()

# Note: choice task 4
# @torch.no_grad()
# def test_multi_output(
#     model: nn.Module,
#     loader: DataLoader,
#     criterion: nn.Module,
#     device: torch.device,
# ):
#     """Test the model on *loader*. Returns (avg_loss, accuracy, predictions, labels)."""
#     model.eval()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     layer_outputs = {'conv1': [], 'conv2': []}
    
#     def hook_conv1(module, input, output):
#         layer_outputs['conv1'].append(output.detach().cpu())
    
#     def hook_conv2(module, input, output):
#         layer_outputs['conv2'].append(output.detach().cpu())
        
#     layer_c1 = model.conv1.register_forward_hook(hook_conv1)
#     layer_c2 = model.conv2.register_forward_hook(hook_conv2)
        
#     for images, labels in loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         running_loss += loss.item() * images.size(0)
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()        
    
#     layer_c1.remove()
#     layer_c2.remove()


#     avg_loss = running_loss / total
#     accuracy = 100.0 * correct / total
#     return avg_loss, accuracy, layer_outputs

def compute_confusion_matrix(predictions, labels):
    
    conf_m = sk.confusion_matrix(predictions.cpu().numpy(), labels.cpu().numpy())
    
    return conf_m

def compute_sne(images):
        
    tsne = skm.TSNE(n_components=2).fit_transform(images)
    
    tsne_x = tsne[:,0]
    tsne_y = tsne[:,1]
    
    tsne_x = scale_value(tsne_x)
    tsne_y = scale_value(tsne_y)
    return tsne_x, tsne_y

# scales the values into a 0-1 range
def scale_value(val):
    range = (np.max(val) - np.min(val))
    
    zero_start = val - np.min(val)
    
    return zero_start / range