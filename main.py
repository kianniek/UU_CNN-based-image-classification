"""
Main entry point for CIFAR-10 CNN training and evaluation.
"""

import os
import json
import argparse
import torch
import json
import os
from datetime import datetime


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
from src.test import test
from src.train import train_model, evaluate
from src.visualize import (
    plot_lr_schedule,
    plot_training_curves,
    plot_augmentation_comparison,
    plot_multi_model_comparison,  # New import
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
        "--batch-size", type=int, default=32, help="Mini-batch size (default: 32) "
        "Standard batch sizes in literature have been: 32, 64, 128, 256"
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
    # Metadata comparison
    parser.add_argument(
        "--compare-models",
        nargs="+",
        help="Paths to metadata JSON files to compare (e.g. results/model1_metadata.json results/model2_metadata.json)",
    )
    # Early stopping / convergence detection
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable automatic convergence detection (early stopping)",
    )
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Epochs without improvement before stopping (default: 3)",
    )
    parser.add_argument(
        "--monitor", type=str, default="val_loss",
        choices=["val_loss", "val_acc"],
        help="Metric to monitor for early stopping (default: val_loss)",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=200,
        help="Max epochs safety cap in automatic mode (default: 200)",
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
    return parser.parse_args()

def save_metadata(args, history, label, save_dir="results"):
    """Saves hyperparameters and training history to a JSON file."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Combine args and history into one serializable dict
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "history": history,
        "label": label
    }
    
    file_path = os.path.join(save_dir, f"{label}_metadata.json")
    with open(file_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Metadata saved → {file_path}")
    return file_path

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
    fixed_history, _ = _run_training(args, augment=augment, device=device, tag="_fixedsplit")
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
        # Grab the model for CIFAR 100 specifically
        model = get_model("cifar100")
        # load cifar100 model weights
        model.load_state_dict(torch.load('results\\cifar100.pth'))
        
        for name, layer in model.named_children():
            # Freezes the first few layers (convolution 1, ReLu, pooling 1)
            
            if name == 'features':
                feature_layers = list(layer.children())
                for sublayer in feature_layers[:-3]:
                    for parameter in sublayer.parameters():
                        parameter.requires_grad = False
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
        early_stopping=args.early_stopping,
        patience=args.patience,
        monitor=args.monitor,
        max_epochs=args.max_epochs,
    )

    # Save metadata for future comparison or graph recreation
    save_metadata(args, history, label)

    # Final test evaluation
    # Softmax is handled by the Cross Entropy loss function
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"Test  Loss {test_loss:.4f}  Acc {test_acc:.2f}%")

    return history, model

def main():
    args = parse_args()

    # If user wants to compare existing runs, do that and exit
    if args.compare_models:
        plot_multi_model_comparison(args.compare_models, metric="val_acc")
        plot_multi_model_comparison(args.compare_models, metric="val_loss")
        return

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

    if args.compare_augmentation:
        history_aug, _ = _run_training(args, augment=True, device=device, tag="_aug")
        history_no_aug, _ = _run_training(args, augment=False, device=device, tag="_noaug")

        plot_augmentation_comparison(history_aug, history_no_aug, model_name=args.model)
        plot_training_curves(history_aug, model_name=f"{args.model}_aug")
        plot_training_curves(history_no_aug, model_name=f"{args.model}_noaug")
        plot_lr_schedule(history_aug["lr"])
    elif args.test_model:
        # single test run
        augment = not args.no_augment
        
        # ---- Run Model on Test Data only (requires pretrained model) ----
        history, _, conf_m, tsne_coord, labels = _run_tests(args, augment=augment, device=device)
        plot_confusion_matrix(conf_m, args.model)
        plot_tsne(tsne_coord, labels, CIFAR10_CLASSES, args.model)

    else:
        augment = not args.no_augment
        history, _ = _run_training(args, augment=augment, device=device)
        plot_lr_schedule(history["lr"])
        plot_training_curves(history, model_name=args.model)

if __name__ == "__main__":
    main()
