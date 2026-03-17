import torch
import numpy as np
from typing import Tuple, List
from sklearn.metrics import confusion_matrix


@torch.no_grad()
def test(
	model: torch.nn.Module,
	loader: torch.utils.data.DataLoader,
	criterion: torch.nn.Module,
	device: torch.device,
) -> Tuple[float, float, np.ndarray]:
	"""Test the model on *loader*.

	Returns
	-------
	avg_loss : float
	accuracy : float (percentage)
	conf_m : np.ndarray (confusion matrix)
	"""
	model.eval()
	running_loss = 0.0
	correct = 0
	total = 0
	preds = []
	labels_all = []

	for images, labels in loader:
		images, labels = images.to(device), labels.to(device)
		outputs = model(images)
		loss = criterion(outputs, labels)

		running_loss += loss.item() * images.size(0)
		_, predicted = outputs.max(1)
		total += labels.size(0)
		correct += predicted.eq(labels).sum().item()

		preds.append(predicted.cpu())
		labels_all.append(labels.cpu())

	preds = torch.cat(preds)
	labels_all = torch.cat(labels_all)

	conf_m = confusion_matrix(labels_all.numpy(), preds.numpy())
	avg_loss = running_loss / total
	accuracy = 100.0 * correct / total
	return avg_loss, accuracy, conf_m