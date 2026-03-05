"""Unit tests for model construction and Kaiming initialisation."""

import torch
import torch.nn as nn
from src.models import SimpleCNN, MediumCNN, DeepCNN, get_model


BATCH = 4
INPUT_SHAPE = (BATCH, 3, 32, 32)


def _check_output_shape(model: nn.Module):
    x = torch.randn(*INPUT_SHAPE)
    out = model(x)
    assert out.shape == (BATCH, 10), f"Expected (4, 10), got {out.shape}"


def _check_kaiming_init(model: nn.Module):
    """Verify that Conv2d/Linear weights are NOT all zeros (i.e. init ran)."""
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            assert m.weight.abs().sum() > 0, f"{name} weights are all zero"
            if m.bias is not None:
                assert m.bias.abs().sum() == 0, f"{name} bias should be zero"


def test_simple_cnn():
    model = SimpleCNN()
    _check_output_shape(model)
    _check_kaiming_init(model)


def test_medium_cnn():
    model = MediumCNN()
    _check_output_shape(model)
    _check_kaiming_init(model)


def test_deep_cnn():
    model = DeepCNN()
    _check_output_shape(model)
    _check_kaiming_init(model)


def test_get_model():
    for name in ("simple", "medium", "deep"):
        model = get_model(name)
        _check_output_shape(model)


if __name__ == "__main__":
    test_simple_cnn()
    test_medium_cnn()
    test_deep_cnn()
    test_get_model()
    print("All model tests passed!")
