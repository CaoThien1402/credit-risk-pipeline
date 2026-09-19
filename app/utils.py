"""Shared utilities for the Streamlit app."""
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.bundle import validate_bundle


def load_model_bundle(path: str) -> dict:
    """Load a joblib model bundle (model + preprocessor + feature list + metadata).

    Validated against model/bundle.py's contract before returning, so a malformed
    bundle fails loudly at app startup rather than partway through a prediction - the
    failure mode that motivated model/bundle.py in the first place. The row-count
    check (validate_bundle's expected_n_rows) is deliberately NOT applied here: it
    needs a live `SELECT COUNT(*) FROM ml_features`, and the app must keep working
    when Postgres is down, since scoring a new application only needs the .pkl.
    tests/test_model_bundles.py covers that check where a DB is guaranteed.
    """
    # Must include the preprocessor, not just the bare model, or the app can't replay
    # the exact input transform used at train time. Expected shape:
    #   {
    #       "model": <trained model>,
    #       "preprocessor": <ColumnTransformer/pipeline used at train time>,
    #       "feature_names": [...],
    #       "trained_at": "...",
    #       "metrics": {"roc_auc": ..., "pr_auc": ...},
    #       "trained_on_n_rows": <int>,
    #   }
    bundle = joblib.load(path)
    validate_bundle(bundle)
    return bundle
