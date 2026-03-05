"""
Data loading utilities for CIFAR-10 and CIFAR-100.

Loads the datasets, applies optional data augmentation, and splits
the training set into a training subset and a validation subset using
an 80/20 ratio (40 k / 10 k images).  The 20 % held-out validation
set is large enough to obtain stable estimates while still leaving
80 % of labelled data for training.
"""

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_cifar10_transforms(augment: bool = False):
    """Return train and test transforms for CIFAR-10.

    When *augment* is True, the training transform includes three data
    augmentation techniques:
        1. Random horizontal flip
        2. Random crop (with padding)
        3. Color jitter (brightness / contrast / saturation)
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    )

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, test_transform


def get_cifar100_transforms(augment: bool = False):
    """Return train and test transforms for CIFAR-100."""
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4865, 0.4409],
        std=[0.2673, 0.2564, 0.2762],
    )

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, test_transform


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_cifar10(
    data_dir: str = "./data",
    batch_size: int = 32,
    val_split: float = 0.2,
    augment: bool = False,
    num_workers: int = 2,
):
    """Load CIFAR-10 and return (train_loader, val_loader, test_loader).

    The official training set (50 000 images) is split 80/20 into a
    training subset (40 000) and a validation subset (10 000).  The
    official test set (10 000) is kept strictly separate and is only
    used for final benchmarking.

    Args:
        data_dir:    Directory where the dataset is stored / downloaded.
        batch_size:  Mini-batch size for all loaders.
        val_split:   Fraction of the training set reserved for validation.
        augment:     Apply data augmentation to the training split.
        num_workers: Number of worker processes for data loading.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_transform, test_transform = get_cifar10_transforms(augment=augment)

    full_train = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    n_val = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    train_subset, val_subset = random_split(
        full_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Apply test transform to the validation subset so it is not augmented.
    val_dataset = _SubsetWithTransform(full_train, val_subset.indices, test_transform)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def load_cifar100(
    data_dir: str = "./data",
    batch_size: int = 32,
    val_split: float = 0.2,
    augment: bool = False,
    num_workers: int = 2,
):
    """Load CIFAR-100 and return (train_loader, val_loader, test_loader).

    Uses coarse (superclass) labels so that the classification task has
    20 output classes, matching the task description.

    Args:
        data_dir:    Directory where the dataset is stored / downloaded.
        batch_size:  Mini-batch size for all loaders.
        val_split:   Fraction of the training set reserved for validation.
        augment:     Apply data augmentation to the training split.
        num_workers: Number of worker processes for data loading.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_transform, test_transform = get_cifar100_transforms(augment=augment)

    full_train = datasets.CIFAR100(
        root=data_dir, train=True, download=True,
        transform=train_transform, target_transform=None,
    )
    _set_coarse_labels(full_train)

    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_transform,
    )
    _set_coarse_labels(test_dataset)

    n_val = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    train_subset, val_subset = random_split(
        full_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    val_dataset = _SubsetWithTransform(full_train, val_subset.indices, test_transform)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Mapping from CIFAR-100 fine class index to coarse superclass index.
# (from the official CIFAR-100 documentation)
_CIFAR100_FINE_TO_COARSE = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]


def _set_coarse_labels(dataset):
    """Replace fine-grained CIFAR-100 labels with the 20 coarse superclass labels."""
    dataset.targets = [_CIFAR100_FINE_TO_COARSE[t] for t in dataset.targets]


class _SubsetWithTransform(torch.utils.data.Dataset):
    """A dataset subset that applies its own transform, overriding the parent."""

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset.data[self.indices[idx]], self.dataset.targets[self.indices[idx]]
        from PIL import Image
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label
