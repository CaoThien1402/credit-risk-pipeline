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
        "trained_on_n_rows": 32581,
    }


def test_valid_bundle_passes():
    validate_bundle(_valid_bundle())


@pytest.mark.parametrize(
    "missing_key",
    ["model", "preprocessor", "feature_names", "trained_at", "metrics", "trained_on_n_rows"],
)
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


@pytest.mark.parametrize("bad_value", [0, -1, "32581", None, 3.5, True])
def test_trained_on_n_rows_must_be_a_positive_int(bad_value):
    bundle = _valid_bundle()
    bundle["trained_on_n_rows"] = bad_value
    with pytest.raises(ValueError, match="trained_on_n_rows"):
        validate_bundle(bundle)


def test_trained_on_n_rows_mismatch_against_expected_is_rejected():
    """The check this exists for: a bundle that claims a row count different from an
    independently-known true count - e.g. a bundle accidentally saved from a
    train/test-split model (~26,064 rows) instead of the full historical dataset
    (32,581 rows)."""
    bundle = _valid_bundle()
    bundle["trained_on_n_rows"] = 26064
    with pytest.raises(ValueError, match="26064.*32581|32581.*26064"):
        validate_bundle(bundle, expected_n_rows=32581)


def test_trained_on_n_rows_match_against_expected_passes():
    bundle = _valid_bundle()
    bundle["trained_on_n_rows"] = 32581
    validate_bundle(bundle, expected_n_rows=32581)


def test_expected_n_rows_not_checked_when_omitted():
    """Callers that don't know the true count (e.g. a pure unit test with no DB) can
    still validate everything else."""
    bundle = _valid_bundle()
    bundle["trained_on_n_rows"] = 999
    validate_bundle(bundle)  # no expected_n_rows given - must not raise
