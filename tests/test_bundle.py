"""Tests for the model-bundle contract app/utils.py::load_model_bundle depends on."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from model.bundle import validate_bundle


def _valid_bundle():
    return {
        "model": object(),
        "preprocessor": object(),
        "feature_names": ["age", "income"],
        "trained_at": "2026-09-12T10:00:00",
        "metrics": {"roc_auc": 0.81, "pr_auc": 0.55},
    }


def test_valid_bundle_passes():
    validate_bundle(_valid_bundle())


@pytest.mark.parametrize("missing_key", ["model", "preprocessor", "feature_names", "trained_at", "metrics"])
def test_missing_key_rejected(missing_key):
    bundle = _valid_bundle()
    del bundle[missing_key]
    with pytest.raises(ValueError, match="missing required keys"):
        validate_bundle(bundle)


def test_bare_model_rejected():
    """The failure mode this contract exists to prevent: saving the model without the
    preprocessor, so app.py can't reproduce the training-time transform."""
    bundle = _valid_bundle()
    bundle["preprocessor"] = None
    with pytest.raises(ValueError, match="preprocessor"):
        validate_bundle(bundle)


def test_empty_feature_names_rejected():
    bundle = _valid_bundle()
    bundle["feature_names"] = []
    with pytest.raises(ValueError, match="feature_names"):
        validate_bundle(bundle)


def test_incomplete_metrics_rejected():
    bundle = _valid_bundle()
    del bundle["metrics"]["pr_auc"]
    with pytest.raises(ValueError, match="pr_auc"):
        validate_bundle(bundle)


def test_non_dict_rejected():
    with pytest.raises(ValueError, match="must be a dict"):
        validate_bundle(["model", "preprocessor"])
