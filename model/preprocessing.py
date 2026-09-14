"""ColumnTransformer + Pipeline used by sessions 9-12.

Reused unchanged by sessions 10-11 (XGBoost swaps only the final estimator) and saved as
`preprocessor` in the session 12 bundle, so app.py can replay the exact training-time
transform on new input.

The two decisions worth being able to defend:

1. **StandardScaler lives inside the Pipeline, not before the split.** `Pipeline.fit`
   only ever sees X_train, so the scaler learns its mean/std from training rows alone.
   Scaling the full frame first and splitting afterwards would let test-set statistics
   into those parameters - the model's evaluation then flatters itself, and nothing in
   the fitted object shows it happened. Enforced by
   tests/test_preprocessing.py::test_notebook_does_not_fit_before_train_test_split.

2. **OneHotEncoder, not LabelEncoder, for nominal categoricals.** LabelEncoder maps
   categories to 0,1,2..., which a linear model reads as order and distance - it would
   infer RENT < OWN < MORTGAGE and that the RENT-to-MORTGAGE gap is twice the RENT-to-OWN
   gap. Neither is true. One-hot gives each category an independent column and imposes no
   ordering. (loan_grade is the deliberate exception: it IS ordinal, A < B < ... < G. It's
   listed in ORDINAL_CATEGORICAL_COLS and currently still one-hot encoded - switching it
   to an OrdinalEncoder is an open session 10-11 decision.)
"""

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """StandardScaler on numeric_cols, OneHotEncoder on categorical_cols."""
    # A column listed twice is transformed twice and silently duplicated in the output
    # matrix - the fit still succeeds and the metrics still look reasonable, so nothing
    # surfaces it. Cheap to rule out here.
    overlap = sorted(set(numeric_cols) & set(categorical_cols))
    if overlap:
        raise ValueError(f"columns listed as both numeric and categorical: {overlap}")
    if not numeric_cols and not categorical_cols:
        raise ValueError("no columns given - the pipeline would train on an empty matrix")

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            # handle_unknown="ignore": the session 13 Streamlit form can submit a category
            # absent from training data (a new loan_intent, say). Raising there would take
            # the app down on valid user input; encoding it as all-zeros degrades that one
            # prediction instead. The tradeoff is that genuinely bad input is scored
            # silently rather than rejected - acceptable here because every categorical
            # field in the form is a closed dropdown, so unknown values mean the training
            # data aged, not that the user typed nonsense.
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        # NOTE: "drop" means a column present in the input but absent from both lists is
        # discarded with no warning - a feature you forgot to classify simply disappears
        # and the metrics still look healthy. ColumnTransformer has no "raise on
        # unlisted" option, so the actual guarantee comes from
        # tests/test_features.py::test_every_feature_column_is_classified, which asserts
        # split_numeric_categorical covers the whole feature set. Don't rely on this line
        # to catch that mistake.
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _default_estimator() -> LogisticRegression:
    return LogisticRegression(
        # ~21.8% positives: without rebalancing, predicting "no default" for everyone
        # already scores 78% accuracy, and the model has little gradient pressure to find
        # defaulters. class_weight rescales the loss rather than resampling, so no
        # synthetic rows enter the training set (SMOTE is evaluated separately in
        # session 11).
        class_weight="balanced",
        # lbfgs defaults to 100 iterations and does not converge on this feature matrix
        # once one-hot expands it.
        max_iter=2000,
        random_state=42,
    )


def build_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    estimator: BaseEstimator | None = None,
    sampler: BaseEstimator | None = None,
) -> Pipeline:
    """Preprocessing + a final estimator as one Pipeline, so preprocessing is fit exactly
    once, on training data only.

    `estimator` defaults to the session 9 Logistic Regression baseline. Session 10-11
    pass an XGBoost/other classifier here instead - the ColumnTransformer built by
    build_preprocessor is reused unchanged, so there is exactly one place that defines
    what "numeric" and "categorical" mean for this dataset, not one copy per notebook.

    `sampler` (session 11: SMOTE) switches the container to imblearn's Pipeline, which is
    the only reason that dependency appears here. The distinction matters: a sampler in an
    imblearn Pipeline runs during `fit` and is bypassed during `predict`/`transform`, so
    synthetic minority rows are created from the training fold only and never appear in
    the data being scored. Resampling before the split - or scoring a test set that SMOTE
    has touched - inflates every metric and is the single most common way SMOTE results
    get reported wrong.

    Sampler position is deliberate: it sits after the preprocessor, because SMOTE
    interpolates numerically and cannot consume raw string categoricals. The tradeoff is
    that it interpolates one-hot columns too, producing fractional values a real one-hot
    row could never have (SMOTENC exists for this, at the cost of bypassing the shared
    ColumnTransformer). Noted rather than silently accepted - see session 11's notebook.
    """
    if estimator is None:
        estimator = _default_estimator()

    steps = [("preprocessor", build_preprocessor(numeric_cols, categorical_cols))]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("classifier", estimator))

    if sampler is None:
        return Pipeline(steps=steps)

    # Imported lazily so the sklearn-only paths don't depend on imblearn being installed.
    from imblearn.pipeline import Pipeline as ImbalancedPipeline

    return ImbalancedPipeline(steps=steps)
