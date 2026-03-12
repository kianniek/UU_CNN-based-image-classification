"""
CNN Models for CIFAR-10 Classification
"""

import torch
import torch.nn as nn
import torch.nn.init as init


# Weight-initialisation helper
# Apply Kaiming (He) uniform initialisation to every Conv2d and Linear layer in model.
def _apply_kaiming_init(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            init.ones_(m.weight)
            init.zeros_(m.bias)

# Model 1 — Simple LeNet-5 CNN (baseline)
# Source: https://www.geeksforgeeks.org/computer-vision/lenet-5-architecture/
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, padding=0, stride=1), # Layer 1: Convolution (1 32x32 => 6 28x28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # Layer 2: Pooling (6 28x28 => 6 14x14)

            nn.Conv2d(6, 16, kernel_size=5, padding=0), # Layer 3: Convolution (6 14x14 => 16 10x10)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # Layer 4: Pooling (16 10x10 => 16 5x5)
            
        )
        self.classifier = nn.Sequential(
            nn.Linear(400, 120), # Layer 5: Fully Connects the flattened layer (16 5x5 => 1 120)
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes), # Layer 6: Output (1 120 => 10 32x32)
        )

        # Apply Kaiming uniform initialisation
        _apply_kaiming_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flattens feature maps
        x = self.classifier(x)
        return x

# Model 2 — leNet-5 with average pooling vs max pooling
# Reasoning: Max pooling always gets the highest value within a kernel, whilst average gets the average of all values in a kernel
class MediumCNN(nn.Module):

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, padding=0, stride=1), # Layer 1: Convolution (1 32x32 => 6 28x28)
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2), # Layer 2: Pooling (6 28x28 => 6 14x14)

            nn.Conv2d(6, 16, kernel_size=5, padding=0), # Layer 3: Convolution (6 14x14 => 16 10x10)
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2), # Layer 4: Pooling (16 10x10 => 16 5x5)
            
        )
        self.classifier = nn.Sequential(
            nn.Linear(400, 120), # Layer 5: Convolution/Fully  Connected (16 5x5 => 1 120)
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes), # Layer 6: Output (1 120 => 10 32x32)
        )

        _apply_kaiming_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flattens feature maps
        x = self.classifier(x)
        return x


# Model 3 — Dropout + AvgPooling
# Reasoning: Dropout is essentially dropping a % of the neurons, prevents overfitting
class DeepCNN(nn.Module):

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, padding=0, stride=1), # Layer 1: Convolution (1 32x32 => 6 28x28)
            nn.ReLU(),
            nn.AvgPool2d(2, 2), # Layer 2: Pooling (6 28x28 => 6 14x14)

            nn.Conv2d(6, 16, kernel_size=5, padding=0), # Layer 3: Convolution (6 14x14 => 16 10x10)
            nn.ReLU(),
            nn.AvgPool2d(2, 2), # Layer 4: Pooling (16 10x10 => 16 5x5) 
            
        )
        self.classifier = nn.Sequential(
            nn.Linear(400, 120), # Layer 5: Convolution/Fully  Connected (16 5x5 => 1 120)
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.Dropout(p=0.2, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes), # Layer 6: Output (1 120 => 10 32x32)
        )

        _apply_kaiming_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Registry helper
MODEL_REGISTRY = {
    "simple": SimpleCNN,
    "medium": MediumCNN,
    "deep": DeepCNN,
}


# """Instantiate a model by name.

# Parameters
# ----------
# name : str
#     One of ``"simple"``, ``"medium"``, ``"deep"``.
# num_classes : int
#     Number of output classes (default 10 for CIFAR-10).

# Returns
# -------
# nn.Module
#     The model with Kaiming-uniform-initialised weights.
# """
def get_model(name: str, num_classes: int = 10) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](num_classes=num_classes)
