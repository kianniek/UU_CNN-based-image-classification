"""
Main entry point for CIFAR-10 CNN training and evaluation.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import numpy as np


from src.data_loader import (
    load_cifar10,
    CIFAR10_CLASSES,
    load_cifar100,
    CIFAR100_SUPERCLASSES,
    get_transforms,
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_MEAN,
    CIFAR100_STD,
)
from src.models import get_model, MODEL_REGISTRY
from src.test import test, test_multi_output
from src.train import train_model, evaluate
from src.visualize import (
    plot_lr_schedule,
    plot_training_curves,
    plot_augmentation_comparison,
    plot_confusion_matrix,
    plot_tsne,
    plot_multiple_out
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate CNN models on CIFAR-10"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="simple",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model architecture to use (default: simple)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Mini-batch size (default: 32)"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="Dataset cache directory"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    # K-fold CV
    parser.add_argument(
        "--kfold",
        type=int,
        default=5,
        help="Enable k-fold cross-validation with specified k (0 to disable)",
    )
    parser.add_argument(
        "--compare-kfold-split",
        action="store_true",
        help="Run both fixed train/val and k-fold CV, then save a side-by-side comparison",
    )

    # Hyperparameter search
    parser.add_argument(
        "--hyperparameter-search",
        action="store_true",
        help="Perform hyperparameter grid search",
    )

    # Choice 1 — LR Scheduler
    parser.add_argument(
        "--no-scheduler",
        action="store_true",
        help="Disable the StepLR scheduler (enabled by default: halve LR every 5 epochs)",
    )
    parser.add_argument(
        "--scheduler-step", type=int, default=5,
        help="StepLR step size in epochs (default: 5)",
    )
    parser.add_argument(
        "--scheduler-gamma", type=float, default=0.5,
        help="StepLR multiplicative factor (default: 0.5 = halve)",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping (enabled by default)",
    )
    # Choice 5 — Data Augmentation comparison
    parser.add_argument(
        "--compare-augmentation",
        action="store_true",
        help="Train twice (with & without augmentation) and plot the comparison",
    )
    # General
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable training-time data augmentation",
    )
    parser.add_argument(
        "--test-model",
        action="store_true",
        help="run test on model with loaded weights",
    )
    parser.add_argument(
        "--multi-output",
        action="store_true",
        help="obtain multiple outputs from the model"
    )
    return parser.parse_args()


def _run_kfold_cv(args, device: torch.device):
    """Run k-fold cross-validation."""
    from src.train import k_fold_cross_validation
    from torchvision import datasets
    from src.models import get_model

    # Load full training dataset
    if args.model == "cifar100":
        from src.data_loader import CIFAR100Super
        full_dataset = CIFAR100Super(
            root=args.data_dir,
            train=True,
            download=True,
            transform=get_transforms(CIFAR100_MEAN, CIFAR100_STD, augment=not args.no_augment),
        )
        num_classes = 20
    else:
        full_dataset = datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            download=True,
            transform=get_transforms(CIFAR10_MEAN, CIFAR10_STD, augment=not args.no_augment),
        )
        num_classes = 10

    def model_factory():
        return get_model(args.model, num_classes=num_classes)

    cv_results = k_fold_cross_validation(
        model_factory,
        full_dataset,
        device,
        k=args.kfold,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=0.0,
        batch_size=args.batch_size,
        seed=args.seed,
        early_stopping=not args.no_early_stopping,
    )

    from src.visualize import plot_kfold_results
    plot_kfold_results(cv_results, model_name=f"{args.model}_k{args.kfold}")

    return cv_results


def _run_hyperparameter_search(args, device: torch.device):
    """Run hyperparameter search."""
    from src.train import hyperparameter_search
    from src.models import get_model

    if args.model == "cifar100":
        train_loader, val_loader, _ = load_cifar100(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            augment_train=not args.no_augment,
            seed=args.seed,
        )
        num_classes = 20
    else:
        train_loader, val_loader, _ = load_cifar10(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            augment_train=not args.no_augment,
            seed=args.seed,
        )
        num_classes = 10

    def model_factory():
        return get_model(args.model, num_classes=num_classes)

    search_results = hyperparameter_search(
        model_factory,
        train_loader,
        val_loader,
        device,
        optimizers=["adam", "sgd", "rmsprop"],
        learning_rates=[1e-3, 1e-4, 1e-5],
        weight_decays=[0.0, 1e-4],
        batch_sizes=[16, 32],
        epochs=args.epochs,
        max_search_points=5,
    )

    from src.visualize import plot_hyperparameter_search
    plot_hyperparameter_search(search_results, model_name=f"{args.model}_hypersearch")

    return search_results


def _run_kfold_vs_fixed_comparison(args, device: torch.device):
    """Run fixed split training and k-fold CV, then save a side-by-side summary."""
    augment = not args.no_augment
    fixed_history, _, _ = _run_training(args, augment=augment, device=device, tag="_fixedsplit")
    cv_results = _run_kfold_cv(args, device)

    comparison = {
        "model": args.model,
        "k": args.kfold,
        "fixed": {
            "best_val_acc": float(max(fixed_history["val_acc"])),
            "best_val_loss": float(min(fixed_history["val_loss"])),
            "final_val_acc": float(fixed_history["val_acc"][-1]),
            "final_val_loss": float(fixed_history["val_loss"][-1]),
            "stopped_epoch": int(fixed_history.get("stopped_epoch", len(fixed_history["val_acc"]))),
        },
        "kfold": {
            "mean_accuracy": float(cv_results["mean_accuracy"]),
            "std_accuracy": float(cv_results["std_accuracy"]),
            "mean_loss": float(cv_results["mean_loss"]),
            "std_loss": float(cv_results["std_loss"]),
        },
    }

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"{args.model}_kfold_vs_fixed_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"K-fold vs fixed comparison saved → {out_path}")

    plot_kfold_vs_fixed_comparison(comparison, model_name=args.model)
    return comparison





def _run_training(args, augment: bool, device: torch.device, tag: str = ""):
    """Helper: load data, build model, train, return history."""
    
    if args.model == "cifar100":
        train_loader, val_loader, test_loader = load_cifar100(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            augment_train=augment,
            seed=args.seed,
        )
    else:
        train_loader, val_loader, test_loader = load_cifar10(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            augment_train=augment,
            seed=args.seed,
        )

    model = get_model(args.model)
    
    if args.model == "finetune":
        # load cifar100 model weights
        model.load_state_dict(torch.load('results\\cifar100.pth'))
        
        for name, layer in model.named_children():
            # Freezes the first few layers (convolution 1, ReLu, pooling 1)
            # could have done this a lot nicer but didn't feel like retraining cifar100 for this 
            
            if name == 'features':
                for parameters in layer[:-3].parameters():
                    parameters.requires_grad = False
            # Changes last layer to 10 outputs
            if name == 'output':
                model.output = nn.Linear(84,10)
                
    model.to(device)
        
    label = f"{args.model}{tag}"
    print(f"\n{'='*60}")
    print(f"Training: {label}  |  augmentation={'ON' if augment else 'OFF'}")
    print(f"{'='*60}")
    print(model)

    history = train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        use_scheduler=not args.no_scheduler,
        early_stopping=not args.no_early_stopping,
        scheduler_step_size=args.scheduler_step,
        scheduler_gamma=args.scheduler_gamma,
        save_path=f"results/{label}_{args.epochs}_{args.lr}.pth",
    )

    # Final test evaluation (use the test helper)
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, conf_m = test(model, test_loader, criterion, device)

    print(f"Test  Loss {test_loss:.4f}  Acc {test_acc:.2f}%")

    return history, model, conf_m

def _run_tests(args, augment: bool, device: torch.device, tag: str = ""):
    """Helper: load data, build model, train, return history."""
    if args.model == "cifar100":
        _, _, test_loader = load_cifar100(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            augment_train=augment,
            seed=args.seed,
        )
    else:
        _, _, test_loader = load_cifar10(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            augment_train=augment,
            seed=args.seed,
        )

    model = get_model(args.model)
    # get model waits for model specified & load them
    
    path_name = os.path.join("results", f'{args.model}.pth')
    weights = torch.load(path_name)
    model.load_state_dict(weights)
    
    model.to(device)
    
    label = f"{args.model}{tag}"
    print(f"\n{'='*60}")
    print(f"Testing: {label}  |  augmentation={'ON' if augment else 'OFF'}")
    print(f"{'='*60}")
    print(model)
    
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, conf_m, tsne_coord, labels = test(model, test_loader, criterion, device)
    print(f"Test  Loss {test_loss:.4f}  Acc {test_acc:.2f}%")

    return test_loss, test_acc, conf_m, tsne_coord, labels

# Separate method that only runs the model on test set
def _run_multi(args, augment: bool, device: torch.device, tag: str = ""):
    """Helper: load data, build model, train, return history."""
    if args.model == "cifar100":
        _, _, test_loader = load_cifar100(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            augment_train=augment,
            seed=args.seed,
        )
    else:
        _, _, test_loader = load_cifar10(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            augment_train=augment,
            seed=args.seed,
        )

    model = get_model(args.model)
    # get model waits for model specified & load them
    
    path_name = os.path.join("results", f'{args.model}.pth')
    weights = torch.load(path_name)
    model.load_state_dict(weights)
    
    model.to(device)
    
    label = f"{args.model}{tag}"
    print(f"\n{'='*60}")
    print(f"Testing: {label}  |  augmentation={'ON' if augment else 'OFF'}")
    print(f"{'='*60}")
    print(model)
    
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, outputs = test_multi_output(model, test_loader, criterion, device)
    print(f"Test  Loss {test_loss:.4f}  Acc {test_acc:.2f}%")

    return test_loss, test_acc, outputs

def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.model == "cifar100":
        print(f"Classes: {CIFAR100_SUPERCLASSES}")
    else:
        print(f"Classes: {CIFAR10_CLASSES}")

    # Priority: k-fold CV, then hyperparameter search, then other modes
    if args.compare_kfold_split:
        _run_kfold_vs_fixed_comparison(args, device)
    elif args.kfold > 0:
        _run_kfold_cv(args, device)
    elif args.hyperparameter_search:
        _run_hyperparameter_search(args, device)
    elif args.compare_augmentation:
        # ---- Choice 5: train with & without augmentation, then compare ----
        history_aug, _, _ = _run_training(args, augment=True, device=device, tag="_aug")
        history_no_aug, _, _ = _run_training(args, augment=False, device=device, tag="_noaug")

        plot_augmentation_comparison(history_aug, history_no_aug, model_name=args.model)
        plot_training_curves(history_aug, model_name=f"{args.model}_aug")
        plot_training_curves(history_no_aug, model_name=f"{args.model}_noaug")

        # Plot LR schedule (same for both runs)
        plot_lr_schedule(history_aug["lr"])
    elif args.test_model:
        # ---- single test run ----
        augment = not args.no_augment
        
        # Note: choice task 4
        if args.multi_output:
            history, _, outputs = _run_multi(args, augment=augment, device= device)
            plot_multiple_out(outputs['conv1'], 'conv1')
            plot_multiple_out(outputs['conv2'], 'conv2')
        else:
            history, _, conf_m, tsne_coord, labels = _run_tests(args, augment=augment, device=device)
            plot_confusion_matrix(conf_m)
            plot_tsne(tsne_coord, labels, CIFAR10_CLASSES)

    else:
        # ---- Normal single training run ----
        augment = not args.no_augment
        history, model, conf_m = _run_training(args, augment=augment, device=device)

        # Choice 1 — Plot LR decay vs. Epochs
        plot_lr_schedule(history["lr"])
        plot_training_curves(history, model_name=args.model)
        plot_confusion_matrix(conf_m, model_name=args.model)


if __name__ == "__main__":
    main()
