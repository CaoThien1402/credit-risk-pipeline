"""The model-bundle contract that app/utils.py::load_model_bundle depends on.

Session 12 saves bundles, app.py (sessions 13-16) loads them. Until this module existed
the contract was only a comment in app/utils.py, so nothing caught a bundle saved with a
missing preprocessor until the Streamlit app failed at runtime.
"""

REQUIRED_KEYS = ("model", "preprocessor", "feature_names", "trained_at", "metrics")
REQUIRED_METRICS = ("roc_auc", "pr_auc")


def validate_bundle(bundle: dict) -> None:
    """Raise ValueError if bundle doesn't match the contract app.py expects."""
    if not isinstance(bundle, dict):
        raise ValueError(f"bundle must be a dict, got {type(bundle).__name__}")

    missing = [k for k in REQUIRED_KEYS if k not in bundle]
    if missing:
        raise ValueError(f"bundle is missing required keys: {missing}")

    if bundle["preprocessor"] is None:
        raise ValueError("bundle['preprocessor'] is None - app.py cannot replay the training transform")

    if not bundle["feature_names"]:
        raise ValueError("bundle['feature_names'] is empty - app.py cannot build the input form")

    missing_metrics = [m for m in REQUIRED_METRICS if m not in bundle["metrics"]]
    if missing_metrics:
        raise ValueError(f"bundle['metrics'] is missing: {missing_metrics}")
