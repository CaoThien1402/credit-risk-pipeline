"""The model-bundle contract that app/utils.py::load_model_bundle depends on.

Session 12 saves bundles, app.py (sessions 13-16) loads them. Until this module existed
the contract was only a comment in app/utils.py, so nothing caught a bundle saved with a
missing preprocessor until the Streamlit app failed at runtime.
"""

REQUIRED_KEYS = ("model", "preprocessor", "feature_names", "trained_at", "metrics", "trained_on_n_rows")
REQUIRED_METRICS = ("roc_auc", "pr_auc")


def validate_bundle(bundle: dict, expected_n_rows: int | None = None) -> None:
    """Raise ValueError if bundle doesn't match the contract app.py expects.

    `expected_n_rows`, when given, checks `bundle["trained_on_n_rows"]` against a
    freshly-queried row count (e.g. `SELECT COUNT(*) FROM ml_features`) instead of just
    checking the field exists. This is the durable version of a check that used to be a
    one-time read of the session 12 notebook's printed output: session 12 refits on
    every historical row rather than the session 10/11 train split, and "the field is
    present and positive" alone can't catch a bundle silently saved from a train-split
    model instead - only comparing it against an independently-obtained count can.
    """
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

    n_rows = bundle["trained_on_n_rows"]
    if not isinstance(n_rows, int) or isinstance(n_rows, bool) or n_rows <= 0:
        raise ValueError(f"bundle['trained_on_n_rows'] must be a positive int, got {n_rows!r}")

    if expected_n_rows is not None and n_rows != expected_n_rows:
        raise ValueError(
            f"bundle claims it was trained on {n_rows} rows, but {expected_n_rows} were "
            "expected - this bundle may have been fit on a subset (e.g. a train/test "
            "split) rather than the full historical dataset"
        )
