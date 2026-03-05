"""
Tests for the CNN architectural study codebase.

Run with:  python -m pytest tests/test_models.py -v
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models import LeNet5Color, Model1BatchNorm, Model2Dropout
from src.train import train_model, train_one_epoch, validate
from src.evaluate import evaluate_model, compute_confusion_matrix
from src.visualize import print_results_table


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def synthetic_loader():
    """Tiny synthetic CIFAR-shaped dataset (32×32 colour images)."""
    x = torch.randn(64, 3, 32, 32)
    y = torch.randint(0, 10, (64,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=16)


@pytest.fixture
def criterion():
    return torch.nn.CrossEntropyLoss()


# ---------------------------------------------------------------------------
# Model architecture tests
# ---------------------------------------------------------------------------

class TestModelShapes:
    @pytest.mark.parametrize("ModelCls", [LeNet5Color, Model1BatchNorm, Model2Dropout])
    def test_output_shape_10(self, ModelCls):
        model = ModelCls(num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert out.shape == (4, 10), f"Expected (4,10), got {out.shape}"

    @pytest.mark.parametrize("ModelCls", [LeNet5Color, Model1BatchNorm, Model2Dropout])
    def test_change_output(self, ModelCls):
        model = ModelCls(num_classes=10)
        model.change_output(20)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 20), f"Expected (2,20), got {out.shape}"

    @pytest.mark.parametrize("ModelCls", [LeNet5Color, Model1BatchNorm, Model2Dropout])
    def test_weights_initialised(self, ModelCls):
        """Kaiming uniform init should produce non-zero weights."""
        model = ModelCls(num_classes=10)
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                assert module.weight.abs().sum().item() > 0

    def test_eval_deterministic(self):
        """Model output should be deterministic in eval mode."""
        model = Model2Dropout(num_classes=10)
        x = torch.randn(8, 3, 32, 32)
        model.eval()
        with torch.no_grad():
            out_a = model(x)
            out_b = model(x)
        assert torch.allclose(out_a, out_b)


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------

class TestTraining:
    def test_train_one_epoch(self, device, synthetic_loader, criterion):
        model = LeNet5Color(num_classes=10).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        metrics = train_one_epoch(model, synthetic_loader, criterion, optimizer, device)
        assert "loss" in metrics and "accuracy" in metrics
        assert metrics["loss"] > 0
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_validate(self, device, synthetic_loader, criterion):
        model = LeNet5Color(num_classes=10).to(device)
        metrics = validate(model, synthetic_loader, criterion, device)
        assert "loss" in metrics and "accuracy" in metrics

    def test_train_model_history_length(self, device, synthetic_loader):
        model = LeNet5Color(num_classes=10).to(device)
        history = train_model(
            model, synthetic_loader, synthetic_loader,
            num_epochs=3, learning_rate=0.001, device=device, verbose=False,
        )
        assert len(history["train_loss"]) == 3
        assert len(history["val_loss"]) == 3
        assert len(history["train_accuracy"]) == 3
        assert len(history["val_accuracy"]) == 3
        assert len(history["lr"]) == 3

    def test_lr_scheduler_reduces_lr(self, device, synthetic_loader):
        """Learning rate should halve every 5 epochs when scheduler is enabled."""
        model = LeNet5Color(num_classes=10).to(device)
        history = train_model(
            model, synthetic_loader, synthetic_loader,
            num_epochs=10, learning_rate=0.001, device=device,
            use_lr_scheduler=True, lr_step_size=5, lr_gamma=0.5,
            verbose=False,
        )
        lr_initial = history["lr"][0]
        assert history["lr"][-1] < lr_initial, "LR should have been reduced"

    def test_loss_decreases_with_training(self, device):
        """Loss should decrease over training on a fixed dataset."""
        torch.manual_seed(42)
        x = torch.randn(128, 3, 32, 32)
        y = torch.randint(0, 10, (128,))
        loader = DataLoader(TensorDataset(x, y), batch_size=32)
        model = LeNet5Color(num_classes=10).to(device)
        history = train_model(
            model, loader, loader, num_epochs=5,
            learning_rate=0.01, device=device, verbose=False,
        )
        assert history["train_loss"][0] > history["train_loss"][-1], \
            "Training loss should decrease over 5 epochs on a fixed dataset."


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------

class TestEvaluation:
    def test_evaluate_model_returns_correct_shape(self, device, synthetic_loader):
        model = LeNet5Color(num_classes=10).to(device)
        acc, targets, preds = evaluate_model(model, synthetic_loader, device)
        assert 0.0 <= acc <= 1.0
        assert len(targets) == len(preds) == 64

    def test_confusion_matrix_shape(self):
        targets = [0, 1, 2, 0, 1, 2]
        preds   = [0, 1, 1, 0, 2, 2]
        cm = compute_confusion_matrix(targets, preds, num_classes=3)
        assert cm.shape == (3, 3)
        assert cm.sum().item() == 6

    def test_confusion_matrix_diagonal(self):
        """Perfect predictions → diagonal matrix."""
        n = 50
        targets = list(range(10)) * 5
        preds = targets[:]
        cm = compute_confusion_matrix(targets, preds, num_classes=10)
        off_diag = cm.sum().item() - cm.diag().sum().item()
        assert off_diag == 0


# ---------------------------------------------------------------------------
# Visualisation helper tests
# ---------------------------------------------------------------------------

class TestVisualise:
    def test_print_results_table(self, capsys):
        results = {
            "CIFAR10_lenet": {"train_acc": 0.75, "val_acc": 0.70, "test_acc": 0.68},
        }
        print_results_table(results)
        captured = capsys.readouterr()
        assert "CIFAR10_lenet" in captured.out
        assert "0.7500" in captured.out


# ---------------------------------------------------------------------------
# Data loader structural tests (no network)
# ---------------------------------------------------------------------------

class TestDataLoaderHelpers:
    def test_cifar100_fine_to_coarse_length(self):
        """The fine-to-coarse mapping must cover all 100 fine classes."""
        from src.data_loader import _CIFAR100_FINE_TO_COARSE
        assert len(_CIFAR100_FINE_TO_COARSE) == 100

    def test_cifar100_coarse_range(self):
        from src.data_loader import _CIFAR100_FINE_TO_COARSE
        coarse = _CIFAR100_FINE_TO_COARSE
        assert min(coarse) == 0
        assert max(coarse) == 19  # 20 superclasses (0–19)
