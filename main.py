"""
Main entry point for the CNN architectural study on CIFAR-10 / CIFAR-100.

Usage
-----
python main.py [--epochs N] [--batch-size B] [--lr LR]
               [--augment] [--lr-scheduler]
               [--data-dir DIR] [--output-dir DIR]
               [--skip-cifar100]

Tasks executed
--------------
1. Load CIFAR-10 (train/val/test splits).
2. Train CIFAR10_lenet  (LeNet-5 baseline).
3. Train CIFAR10_model1 (+ Batch Normalisation).
4. Train CIFAR10_model2 (+ Batch Norm + Dropout).
5. Load CIFAR-100 (coarse labels, 20 classes), train best architecture.
6. Fine-tune CIFAR100_model on CIFAR-10 (CIFAR10_pretrained).
7. Evaluate CIFAR10_model2 and CIFAR10_pretrained on the test set.
8. Save weights, plots, and summary table.

Choice tasks included
---------------------
CHOICE 1 – Learning rate scheduler (--lr-scheduler flag).
CHOICE 5 – Data augmentation (--augment flag).
"""

import argparse
import os
import sys

import torch

from src.data_loader import load_cifar10, load_cifar100
from src.evaluate import compute_confusion_matrix, evaluate_model
from src.models import LeNet5Color, Model1BatchNorm, Model2Dropout
from src.train import train_model
from src.visualize import (
    plot_confusion_matrix,
    plot_lr_curve,
    plot_training_curves,
    print_results_table,
)

# CIFAR-10 class names (used for confusion matrix labels)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# CIFAR-100 coarse (superclass) names
CIFAR100_COARSE_CLASSES = [
    "aquatic mammals", "fish", "flowers", "food containers",
    "fruit and vegetables", "household electrical devices",
    "household furniture", "insects", "large carnivores",
    "large man-made outdoor things", "large natural outdoor scenes",
    "large omnivores and herbivores", "medium-sized mammals",
    "non-insect invertebrates", "people", "reptiles",
    "small mammals", "trees", "vehicles 1", "vehicles 2",
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="CNN study: CIFAR-10 / CIFAR-100"
    )
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs (default: 30).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Mini-batch size (default: 32).")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial Adam learning rate (default: 0.001).")
    parser.add_argument("--augment", action="store_true",
                        help="Apply data augmentation (CHOICE 5).")
    parser.add_argument("--lr-scheduler", action="store_true",
                        help="Halve LR every 5 epochs (CHOICE 1).")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Directory for dataset storage.")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Directory for saved weights and plots.")
    parser.add_argument("--skip-cifar100", action="store_true",
                        help="Skip CIFAR-100 training and fine-tuning tasks.")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader worker processes (default: 2).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  ✓ Weights saved → {path}")


def _load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model


def _train_and_record(
    name, model, train_loader, val_loader,
    num_epochs, lr, device, use_lr_scheduler, output_dir, augment_label
):
    """Train a model, plot curves, and return the history dict."""
    print(f"\n{'='*60}")
    print(f"  Training {name}{augment_label}")
    print(f"{'='*60}")
    model.to(device)
    history = train_model(
        model, train_loader, val_loader,
        num_epochs=num_epochs,
        learning_rate=lr,
        device=device,
        use_lr_scheduler=use_lr_scheduler,
    )
    plot_training_curves(
        history, name,
        save_path=os.path.join(output_dir, f"{name}_curves.png"),
    )
    if use_lr_scheduler:
        plot_lr_curve(
            history, name,
            save_path=os.path.join(output_dir, f"{name}_lr.png"),
        )
    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Epochs: {args.epochs}  |  Batch size: {args.batch_size}  "
          f"|  LR: {args.lr}  |  Augment: {args.augment}  "
          f"|  LR scheduler: {args.lr_scheduler}\n")

    augment_label = " (+ augmentation)" if args.augment else ""

    # ------------------------------------------------------------------
    # TASK 1 – Load CIFAR-10
    # ------------------------------------------------------------------
    print("Loading CIFAR-10 …")
    train_loader, val_loader, test_loader = load_cifar10(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=0.2,
        augment=args.augment,
        num_workers=args.num_workers,
    )
    print(f"  Train batches: {len(train_loader)}  "
          f"Val batches: {len(val_loader)}  "
          f"Test batches: {len(test_loader)}")

    # ------------------------------------------------------------------
    # TASK 2 – Train three CIFAR-10 models
    # ------------------------------------------------------------------
    results = {}

    # ---- CIFAR10_lenet ----
    lenet = LeNet5Color(num_classes=10)
    history_lenet = _train_and_record(
        "CIFAR10_lenet", lenet, train_loader, val_loader,
        args.epochs, args.lr, device, args.lr_scheduler,
        args.output_dir, augment_label,
    )
    _save_model(lenet, os.path.join(args.output_dir, "CIFAR10_lenet.pth"))
    results["CIFAR10_lenet"] = {
        "train_acc": history_lenet["train_accuracy"][-1],
        "val_acc":   history_lenet["val_accuracy"][-1],
    }

    # ---- CIFAR10_model1 ----
    model1 = Model1BatchNorm(num_classes=10)
    history_model1 = _train_and_record(
        "CIFAR10_model1", model1, train_loader, val_loader,
        args.epochs, args.lr, device, args.lr_scheduler,
        args.output_dir, augment_label,
    )
    _save_model(model1, os.path.join(args.output_dir, "CIFAR10_model1.pth"))
    results["CIFAR10_model1"] = {
        "train_acc": history_model1["train_accuracy"][-1],
        "val_acc":   history_model1["val_accuracy"][-1],
    }

    # ---- CIFAR10_model2 ----
    model2 = Model2Dropout(num_classes=10)
    history_model2 = _train_and_record(
        "CIFAR10_model2", model2, train_loader, val_loader,
        args.epochs, args.lr, device, args.lr_scheduler,
        args.output_dir, augment_label,
    )
    _save_model(model2, os.path.join(args.output_dir, "CIFAR10_model2.pth"))
    results["CIFAR10_model2"] = {
        "train_acc": history_model2["train_accuracy"][-1],
        "val_acc":   history_model2["val_accuracy"][-1],
    }

    # ------------------------------------------------------------------
    # TASK 3 – Evaluate three models on test set (quick check)
    # ------------------------------------------------------------------
    print("\nEvaluating CIFAR-10 models on test set …")
    for name, model in [("CIFAR10_lenet", lenet),
                         ("CIFAR10_model1", model1),
                         ("CIFAR10_model2", model2)]:
        acc, _, _ = evaluate_model(model, test_loader, device)
        results[name]["test_acc"] = acc
        print(f"  {name}: Test Acc = {acc:.4f}")

    # ------------------------------------------------------------------
    # Identify best model (by validation accuracy)
    # ------------------------------------------------------------------
    best_name = max(
        ["CIFAR10_lenet", "CIFAR10_model1", "CIFAR10_model2"],
        key=lambda n: results[n]["val_acc"],
    )
    best_model_map = {
        "CIFAR10_lenet":  (LeNet5Color, lenet),
        "CIFAR10_model1": (Model1BatchNorm, model1),
        "CIFAR10_model2": (Model2Dropout, model2),
    }
    best_cls, best_model = best_model_map[best_name]
    print(f"\n  Best model (by val acc): {best_name}  "
          f"(Val Acc = {results[best_name]['val_acc']:.4f})")

    if args.skip_cifar100:
        print("\nSkipping CIFAR-100 tasks (--skip-cifar100).")
        print_results_table(results)
        return

    # ------------------------------------------------------------------
    # TASK 4 – Train on CIFAR-100 (20 coarse classes)
    # ------------------------------------------------------------------
    print("\nLoading CIFAR-100 (coarse labels, 20 classes) …")
    train100_loader, val100_loader, test100_loader = load_cifar100(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=0.2,
        augment=args.augment,
        num_workers=args.num_workers,
    )

    cifar100_model = best_cls(num_classes=10)
    cifar100_model.change_output(20)
    cifar100_model.to(device)

    print(f"\n{'='*60}")
    print(f"  Training CIFAR100_model ({best_name} architecture, 20 classes)")
    print(f"{'='*60}")
    history_cifar100 = train_model(
        cifar100_model, train100_loader, val100_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        use_lr_scheduler=args.lr_scheduler,
    )
    plot_training_curves(
        history_cifar100, "CIFAR100_model",
        save_path=os.path.join(args.output_dir, "CIFAR100_model_curves.png"),
    )
    _save_model(cifar100_model, os.path.join(args.output_dir, "CIFAR100_model.pth"))
    results["CIFAR100_model"] = {
        "train_acc": history_cifar100["train_accuracy"][-1],
        "val_acc":   history_cifar100["val_accuracy"][-1],
        "test_acc":  float("nan"),
    }

    # ------------------------------------------------------------------
    # TASK 5 – Fine-tune CIFAR100_model on CIFAR-10 (CIFAR10_pretrained)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Fine-tuning CIFAR10_pretrained (CIFAR-100 → CIFAR-10)")
    print(f"{'='*60}")
    pretrained_model = best_cls(num_classes=10)
    # Copy all weights from CIFAR-100 model, then replace the output head.
    pretrained_model.load_state_dict(cifar100_model.state_dict(), strict=False)
    pretrained_model.change_output(10)
    pretrained_model.to(device)

    fine_tune_lr = args.lr / 2.0
    print(f"  Fine-tune LR: {fine_tune_lr}")
    history_pretrained = train_model(
        pretrained_model, train_loader, val_loader,
        num_epochs=args.epochs,
        learning_rate=fine_tune_lr,
        device=device,
        use_lr_scheduler=args.lr_scheduler,
    )
    plot_training_curves(
        history_pretrained, "CIFAR10_pretrained",
        save_path=os.path.join(args.output_dir, "CIFAR10_pretrained_curves.png"),
    )
    _save_model(pretrained_model, os.path.join(args.output_dir, "CIFAR10_pretrained.pth"))
    results["CIFAR10_pretrained"] = {
        "train_acc": history_pretrained["train_accuracy"][-1],
        "val_acc":   history_pretrained["val_accuracy"][-1],
    }

    # ------------------------------------------------------------------
    # TASK 6 – Test best scratch model vs. pretrained model
    # ------------------------------------------------------------------
    print("\nFinal evaluation on CIFAR-10 test set …")

    # Best scratch model (already evaluated above, just add confusion matrix)
    acc_scratch, targets_scratch, preds_scratch = evaluate_model(
        best_model, test_loader, device
    )
    results[best_name]["test_acc"] = acc_scratch
    cm_scratch = compute_confusion_matrix(targets_scratch, preds_scratch, 10)
    plot_confusion_matrix(
        cm_scratch, CIFAR10_CLASSES, best_name,
        save_path=os.path.join(args.output_dir, f"{best_name}_confusion.png"),
    )

    # Pretrained model
    acc_pretrained, targets_pretrained, preds_pretrained = evaluate_model(
        pretrained_model, test_loader, device
    )
    results["CIFAR10_pretrained"]["test_acc"] = acc_pretrained
    cm_pretrained = compute_confusion_matrix(targets_pretrained, preds_pretrained, 10)
    plot_confusion_matrix(
        cm_pretrained, CIFAR10_CLASSES, "CIFAR10_pretrained",
        save_path=os.path.join(args.output_dir, "CIFAR10_pretrained_confusion.png"),
    )

    print(f"  {best_name} (scratch)   Test Acc: {acc_scratch:.4f}")
    print(f"  CIFAR10_pretrained      Test Acc: {acc_pretrained:.4f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n")
    print_results_table(results)
    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
