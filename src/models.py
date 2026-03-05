"""
CNN model architectures for the CIFAR classification study.

Models
------
LeNet5Color
    LeNet-5 adapted for 3-channel 32×32 color input (baseline).

Model1BatchNorm
    LeNet5Color + Batch Normalisation after every convolutional layer.
    Batch normalisation accelerates training and acts as a regulariser,
    which we expect to improve generalisation over the baseline.

Model2Dropout
    Model1BatchNorm + Dropout (p=0.5) applied after each fully-connected
    hidden layer.  Dropout is a complementary regulariser that reduces
    co-adaptation of neurons and further improves test-set performance.

All models use:
    - ReLU activations for hidden layers
    - Softmax on the output layer (for interpretability; the training code
      uses CrossEntropyLoss which expects raw logits, so Softmax is applied
      only during inference / evaluation)
    - kaiming_uniform weight initialisation
    - MaxPool2d pooling layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Weight initialisation helper
# ---------------------------------------------------------------------------

def _init_weights(module: nn.Module):
    """Apply Kaiming Uniform initialisation to Conv and Linear layers."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# Baseline: LeNet-5 for color images
# ---------------------------------------------------------------------------

class LeNet5Color(nn.Module):
    """LeNet-5 adapted for 3-channel 32×32 images (CIFAR-10 baseline).

    Architecture (no zero-padding, MaxPool):
        Input  : 3 × 32 × 32
        Conv1  : 6 filters, 5×5  → 6 × 28 × 28  + ReLU
        Pool1  : MaxPool 2×2      → 6 × 14 × 14
        Conv2  : 16 filters, 5×5  → 16 × 10 × 10 + ReLU
        Pool2  : MaxPool 2×2      → 16 × 5 × 5
        Conv3  : 120 filters, 5×5 → 120 × 1 × 1  + ReLU  (C5 in original)
        FC1    : 120 → 84         + ReLU
        FC2    : 84  → num_classes (logits)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),       # 6×28×28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 6×14×14
            nn.Conv2d(6, 16, kernel_size=5),       # 16×10×10
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16×5×5
            nn.Conv2d(16, 120, kernel_size=5),     # 120×1×1
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def change_output(self, num_classes: int):
        """Replace the final linear layer to support a different class count."""
        in_features = self.classifier[-1].in_features
        self.classifier[-1] = nn.Linear(in_features, num_classes)
        nn.init.kaiming_uniform_(self.classifier[-1].weight, nonlinearity="relu")
        nn.init.zeros_(self.classifier[-1].bias)


# ---------------------------------------------------------------------------
# Variant 1: LeNet-5 + Batch Normalisation
# ---------------------------------------------------------------------------

class Model1BatchNorm(nn.Module):
    """LeNet5Color with Batch Normalisation after each convolutional layer.

    Change from baseline: Batch Normalisation (BN) is inserted after each
    Conv2d + ReLU pair.  BN normalises activations across the mini-batch,
    which stabilises training, allows higher learning rates in practice,
    and provides mild regularisation.  We expect faster convergence and
    improved validation accuracy compared with the plain LeNet-5 baseline.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.BatchNorm2d(120),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def change_output(self, num_classes: int):
        in_features = self.classifier[-1].in_features
        self.classifier[-1] = nn.Linear(in_features, num_classes)
        nn.init.kaiming_uniform_(self.classifier[-1].weight, nonlinearity="relu")
        nn.init.zeros_(self.classifier[-1].bias)


# ---------------------------------------------------------------------------
# Variant 2: LeNet-5 + Batch Normalisation + Dropout
# ---------------------------------------------------------------------------

class Model2Dropout(nn.Module):
    """Model1BatchNorm with Dropout added after each fully-connected hidden layer.

    Change from Model 1: Dropout (p=0.5) is added after each hidden FC layer.
    Dropout randomly sets activations to zero during training, preventing
    co-adaptation of neurons and reducing overfitting.  This is the most
    commonly reported improvement for FC-heavy architectures and should
    further improve generalisation, especially when the validation accuracy
    of Model 1 saturates.
    """

    def __init__(self, num_classes: int = 10, dropout_p: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.BatchNorm2d(120),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(84, num_classes),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def change_output(self, num_classes: int):
        in_features = self.classifier[-1].in_features
        self.classifier[-1] = nn.Linear(in_features, num_classes)
        nn.init.kaiming_uniform_(self.classifier[-1].weight, nonlinearity="relu")
        nn.init.zeros_(self.classifier[-1].bias)
