"""Shared utilities for the Streamlit app."""
import joblib


def load_model_bundle(path: str) -> dict:
    """Load a joblib model bundle (model + preprocessor + feature list + metadata)."""
    # Must include the preprocessor, not just the bare model, or the app can't replay
    # the exact input transform used at train time. Expected shape:
    #   {
    #       "model": <trained model>,
    #       "preprocessor": <ColumnTransformer/pipeline used at train time>,
    #       "feature_names": [...],
    #       "trained_at": "...",
    #       "metrics": {"roc_auc": ..., "pr_auc": ...},
    #   }
    return joblib.load(path)
