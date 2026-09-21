"""Integration check for the real models/*.pkl artifacts produced by
notebooks/05_explainability_and_bundles.ipynb.

Skips if either the bundle file or a reachable Postgres is missing. models/*.pkl is
gitignored and never present in CI, so this is a local-only verification to run after
regenerating the bundles - not a CI gate. Its job is the one thing a bundle-shaped dict
in a unit test can never prove: that the *actual saved file* was really trained on every
historical row, checked against a live, independently-run COUNT(*).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "etl"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import pytest
from sqlalchemy import text

from config import get_engine
from model.bundle import validate_bundle

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def _bundle_path(filename):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        pytest.skip(f"{filename} not present - run notebooks/05_explainability_and_bundles.ipynb first")
    return path


@pytest.fixture(scope="module")
def db_engine():
    """A connected engine, or skip. Every test in this module needs the database, so the
    skip lives here rather than inside one of them - a test that calls get_engine()
    itself would raise OperationalError and report as a FAILURE, not a skip, the moment
    Docker is down. That is exactly what happened before this fixture existed, and it
    contradicted this module's own docstring."""
    try:
        engine = get_engine()
        with engine.connect():
            pass
    except Exception as exc:
        pytest.skip(f"no reachable Postgres ({exc}) - run `docker compose up -d` first")
    return engine


@pytest.fixture(scope="module")
def historical_row_count(db_engine):
    with db_engine.connect() as conn:
        return conn.execute(text("SELECT COUNT(*) FROM ml_features")).scalar()


@pytest.mark.parametrize("filename", ["at_application_model.pkl", "portfolio_risk_model.pkl"])
def test_bundle_was_trained_on_the_full_historical_dataset(filename, historical_row_count):
    """Guards the refit-on-100%-historical decision (see CLAUDE.md). Catches a bundle
    silently saved from a train/test-split model (~26,064 rows) instead of a refit on
    every historical row - independently of whatever the notebook printed when it ran."""
    bundle = joblib.load(_bundle_path(filename))
    validate_bundle(bundle, expected_n_rows=historical_row_count)


@pytest.mark.parametrize("filename", ["at_application_model.pkl", "portfolio_risk_model.pkl"])
def test_bundle_predicts_through_preprocessor_then_model(filename, db_engine):
    """Contract check on the real artifact: preprocessor.transform -> model.predict_proba,
    the same two explicit steps app.py will call - not the whole fitted Pipeline object."""
    import pandas as pd

    bundle = joblib.load(_bundle_path(filename))
    sample = pd.read_sql(
        f"SELECT {', '.join(bundle['feature_names'])} FROM ml_features LIMIT 1", db_engine
    )
    transformed = bundle["preprocessor"].transform(sample)
    proba = bundle["model"].predict_proba(transformed)
    assert proba.shape == (1, 2)
