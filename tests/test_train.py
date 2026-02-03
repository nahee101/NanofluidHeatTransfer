"""Tests for the training module."""
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import get_models, MODEL_NAME


def test_get_models_returns_dict():
    """Test that get_models returns a dictionary."""
    models = get_models()
    assert isinstance(models, dict)


def test_get_models_has_three_models():
    """Test that get_models returns exactly 3 models."""
    models = get_models()
    assert len(models) == 3


def test_get_models_contains_expected_models():
    """Test that get_models contains the expected model types."""
    models = get_models()
    expected_models = ['RandomForest', 'GradientBoosting', 'Ridge']
    for model_name in expected_models:
        assert model_name in models, f"{model_name} not found in models"


def test_models_have_fit_method():
    """Test that all models have a fit method."""
    models = get_models()
    for name, model in models.items():
        assert hasattr(model, 'fit'), f"{name} does not have fit method"


def test_models_have_predict_method():
    """Test that all models have a predict method."""
    models = get_models()
    for name, model in models.items():
        assert hasattr(model, 'predict'), f"{name} does not have predict method"


def test_model_name_is_defined():
    """Test that MODEL_NAME constant is defined."""
    assert MODEL_NAME is not None
    assert isinstance(MODEL_NAME, str)
    assert len(MODEL_NAME) > 0
