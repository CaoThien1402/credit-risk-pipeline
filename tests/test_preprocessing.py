"""Guards the train-split discipline in model/preprocessing.py.

The failure this exists to catch: fitting the preprocessor on the full dataset before
train_test_split. That leaks test-set statistics into training and inflates the metrics,
and it leaves no trace at runtime - the code still runs, the numbers just get better for
the wrong reason.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from model.preprocessing import build_pipeline

NUMERIC_COLS = ["age", "income"]
CATEGORICAL_COLS = ["home_ownership"]


def _synthetic_split():
    """Train and test rows with deliberately different means, so a scaler fitted on the
    full frame is numerically distinguishable from one fitted on train alone."""
    train = pd.DataFrame({
        "age": [25, 30, 35, 40] * 5,
        "income": [50_000.0, 60_000.0, 70_000.0, 80_000.0] * 5,
        "home_ownership": ["RENT", "OWN", "MORTGAGE", "RENT"] * 5,
        "loan_status": [0, 1, 0, 1] * 5,
    })
    test = pd.DataFrame({
        "age": [90, 95] * 5,
        "income": [900_000.0, 950_000.0] * 5,
        "home_ownership": ["RENT", "OWN"] * 5,
        "loan_status": [0, 1] * 5,
    })
    return train, test


def _find_fitted_scaler(estimator):
    """Locate the fitted StandardScaler regardless of how the pipeline is structured."""
    if isinstance(estimator, StandardScaler) and hasattr(estimator, "mean_"):
        return estimator
    for attr in ("steps", "transformers_", "transformers"):
        for entry in getattr(estimator, attr, []) or []:
            candidate = entry[1] if isinstance(entry, tuple) else entry
            found = _find_fitted_scaler(candidate)
            if found is not None:
                return found
    return None


def _build():
    return build_pipeline(NUMERIC_COLS, CATEGORICAL_COLS)


def test_scaler_is_inside_the_pipeline_and_learns_train_statistics():
    """Catches a missing scaler, a scaler wired to the wrong columns, or scaling done
    outside the Pipeline.

    Note it deliberately does NOT claim to catch a scaler pre-fitted on the full dataset:
    ColumnTransformer.fit clones and re-fits its transformers, so passing in a pre-fitted
    scaler has no effect. Ordering is enforced by
    test_notebook_does_not_fit_before_train_test_split instead.
    """
    train, _ = _synthetic_split()
    pipeline = _build()
    pipeline.fit(train[NUMERIC_COLS + CATEGORICAL_COLS], train["loan_status"])

    scaler = _find_fitted_scaler(pipeline)
    assert scaler is not None, "no fitted StandardScaler found inside the pipeline"

    train_means = train[NUMERIC_COLS].mean().to_numpy()
    np.testing.assert_allclose(
        np.sort(scaler.mean_), np.sort(train_means), rtol=1e-6,
        err_msg="scaler statistics don't match the data it was fitted on",
    )


def test_pipeline_does_not_refit_on_transform():
    """predict on unseen data must not update the fitted statistics."""
    train, test = _synthetic_split()
    pipeline = _build()
    pipeline.fit(train[NUMERIC_COLS + CATEGORICAL_COLS], train["loan_status"])

    before = _find_fitted_scaler(pipeline).mean_.copy()
    pipeline.predict_proba(test[NUMERIC_COLS + CATEGORICAL_COLS])
    after = _find_fitted_scaler(pipeline).mean_

    np.testing.assert_allclose(before, after, rtol=1e-12)


def test_unseen_category_does_not_crash_prediction():
    """A new-application form can submit a category absent from the training data;
    OneHotEncoder must be configured to tolerate it rather than raise."""
    train, _ = _synthetic_split()
    pipeline = _build()
    pipeline.fit(train[NUMERIC_COLS + CATEGORICAL_COLS], train["loan_status"])

    unseen = pd.DataFrame({"age": [33], "income": [65_000.0], "home_ownership": ["OTHER"]})
    pipeline.predict_proba(unseen)


# --- notebook ordering guard -------------------------------------------------------
# The leak a pipeline-level test cannot see: scaling/encoding the full frame and only
# then calling train_test_split. Nothing about the fitted pipeline reveals it, because
# the damage is already baked into the data handed to fit(). The only reliable signal is
# the order the calls appear in the notebook source.

import json
import re

NOTEBOOK = os.path.join(os.path.dirname(__file__), "..", "notebooks", "02_baseline_model.ipynb")

FIT_CALL = re.compile(r"\.fit(?:_transform)?\s*\(")
SPLIT_CALL = re.compile(r"train_test_split\s*\(")


def _code_lines(notebook_path):
    """Flatten a notebook's code cells into (cell_index, line_no, text), comments stripped."""
    with open(notebook_path, encoding="utf-8") as fh:
        nb = json.load(fh)
    out = []
    for cell_idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        for line_no, raw in enumerate(cell["source"], 1):
            code = raw.split("#", 1)[0]
            if code.strip():
                out.append((cell_idx, line_no, code))
    return out


def test_notebook_does_not_fit_before_train_test_split():
    lines = _code_lines(NOTEBOOK)
    split_positions = [i for i, (_, _, code) in enumerate(lines) if SPLIT_CALL.search(code)]
    assert split_positions, "no train_test_split call found in the training notebook"
    first_split = split_positions[0]

    premature = [
        (cell_idx, line_no, code.strip())
        for i, (cell_idx, line_no, code) in enumerate(lines)
        if i < first_split and FIT_CALL.search(code)
    ]
    assert not premature, (
        "fit()/fit_transform() called before train_test_split - this leaks test-set "
        f"statistics into training: {premature}"
    )


def test_column_in_both_lists_is_rejected():
    """A duplicated column is transformed twice and silently widens the matrix - the fit
    succeeds and the metrics look normal, so nothing surfaces it."""
    from model.preprocessing import build_preprocessor
    with pytest.raises(ValueError, match="both numeric and categorical"):
        build_preprocessor(["age", "income"], ["income"])


def test_empty_column_lists_are_rejected():
    from model.preprocessing import build_preprocessor
    with pytest.raises(ValueError, match="no columns given"):
        build_preprocessor([], [])
