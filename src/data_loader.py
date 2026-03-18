"""
CIFAR-10 Data Loader
====================
Downloads and prepares the CIFAR-10 dataset with the following splits:

- **Training set** : 80 % of the original 50 000 training images → 40 000
- **Validation set**: 20 % of the original 50 000 training images → 10 000
- **Test set**     : the full 10 000 test images (used only for final benchmarking)

Rationale for the 80/20 split
-----------------------------
An 80/20 train/validation split is a well-established default that balances two
competing concerns:

1. *Sufficient training data* – 40 000 images still provide ample variety for
    learning robust features on CIFAR-10's 10 classes.
2. *Reliable validation estimates* – 10 000 validation samples (1 000 per class
    when stratified) yield tight confidence intervals on accuracy, making it
    practical to compare architectural and hyper-parameter choices during
    development without touching the held-out test set.

A more aggressive split (e.g. 90/10) would leave only 5 000 validation samples
and increase the variance of the validation metric, while a 70/30 split would
unnecessarily reduce the training set size for a dataset of this scale.
"""

import os
import pickle
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, transforms


# Default num_workers: 0 on Windows (spawn-based multiprocessing is fragile),
# 2 elsewhere (fork is safe).
_DEFAULT_WORKERS = 2 if os.name == "nt" else 2


# CIFAR-10 channel-wise statistics (pre-computed over the training set)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# CIFAR-10 class names (index → human-readable label)
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


# Return a composed transform pipeline.
#
# When ``augment=True`` the following **4 augmentation techniques** are applied1
# (satisfies the ≥ 3 requirement for Choice 5):
#   1. RandomCrop with 4-pixel padding
#   2. RandomHorizontalFlip
#   3. ColorJitter (brightness, contrast, saturation)
#   4. RandomRotation (±15°)

# edit: adjusted function to take in mean and std as arguments
def get_transforms(mean, std, augment: bool = False):

    if augment:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2
                ),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


# Download CIFAR-10 and return DataLoaders for train / val / test.
def load_cifar10(
    data_dir: str = "./data",
    batch_size: int = 32,
    num_workers: int = _DEFAULT_WORKERS,
    val_fraction: float = 0.2,
    augment_train: bool = True,
    seed: int = 42,
):

    full_train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(CIFAR10_MEAN, CIFAR10_STD, augment=augment_train),
        
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=get_transforms(CIFAR10_MEAN, CIFAR10_STD, augment=False),
    )

    total_train = len(full_train_dataset)
    val_size = int(total_train * val_fraction)
    train_size = total_train - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )

    full_train_eval = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=get_transforms(CIFAR10_MEAN, CIFAR10_STD, augment=False),
    )
    val_dataset = torch.utils.data.Subset(full_train_eval, val_dataset.indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(
        f"CIFAR-10 loaded — "
        f"train: {len(train_dataset)}, "
        f"val: {len(val_dataset)}, "
        f"test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader


# Default num_workers: 0 on Windows (spawn-based multiprocessing is fragile),
# 2 elsewhere (fork is safe).
_DEFAULT_WORKERS = 0 if os.name == "nt" else 2


# CIFAR-10 channel-wise statistics (pre-computed over the training set)
CIFAR100_MEAN = (0.5070, 0.4865,  0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2761)

# CIFAR-100 class names (index → human-readable label)
CIFAR100_SUPERCLASSES = (
    "aquatic_mammals",
    "fish",
    "flowers",
    "food_containers",
    "fruit_and_vegetables",
    "household_electrical_devices",
    "household_furniture",
    "insects",
    "large_carnivores",
    "large_man_made_outdoor_things",
    "large_natural_outdoor_scenes",
    "large_omnivores_and_herbivores",
    "medium_mammals",
    "non_insect_invertebrates",
    "people",
    "reptiles",
    "small_mammals",
    "trees",
    "vehicles_1",
    "vehicles_2",
)

class CIFAR100Super(datasets.CIFAR100):
    def __init__(self, *args, **kwargs):
        super(CIFAR100Super, self).__init__(*args, **kwargs)

        file_name = "train" if self.train else "test"
        file_path = os.path.join(self.root, self.base_folder, file_name)

        with open(file_path, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
            self.coarse_targets = entry["coarse_labels"]
        
    def __getitem__(self,index):
        img, _ = super().__getitem__(index)
        target = self.coarse_targets[index]
        return img, target

# Download CIFAR-100 and return DataLoaders for train / val / test.
def load_cifar100(
    data_dir: str = "./data",
    batch_size: int = 32,
    num_workers: int = _DEFAULT_WORKERS,
    val_fraction: float = 0.2,
    augment_train: bool = True,
    seed: int = 42,
):

    full_train_dataset = CIFAR100Super(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(CIFAR100_MEAN, CIFAR100_STD, augment=augment_train),
    )

    test_dataset =  CIFAR100Super(
        root=data_dir,
        train=False,
        download=True,
        transform=get_transforms(CIFAR100_MEAN, CIFAR100_STD, augment=False),
    )

    total_train = len(full_train_dataset)
    val_size = int(total_train * val_fraction)
    train_size = total_train - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )

    full_train_eval =  CIFAR100Super(
        root=data_dir,
        train=True,
        download=False,
        transform=get_transforms(CIFAR100_MEAN, CIFAR100_STD, augment=False),
    )
    val_dataset = torch.utils.data.Subset(full_train_eval, val_dataset.indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(
        f"CIFAR-100 loaded — "
        f"train: {len(train_dataset)}, "
        f"val: {len(val_dataset)}, "
        f"test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader


