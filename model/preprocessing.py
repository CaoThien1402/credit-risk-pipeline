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

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """StandardScaler on numeric_cols, OneHotEncoder on categorical_cols."""
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
        # Fail loudly if a column reaches this that belongs to neither list, rather than
        # silently dropping it - a dropped feature is invisible in the metrics.
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    """Preprocessing + Logistic Regression baseline as one estimator, so preprocessing is
    fit exactly once, on training data only."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
            (
                "classifier",
                LogisticRegression(
                    # ~21.8% positives: without rebalancing, predicting "no default" for
                    # everyone already scores 78% accuracy, and the model has little
                    # gradient pressure to find defaulters. class_weight rescales the loss
                    # rather than resampling, so no synthetic rows enter the training set
                    # (SMOTE is evaluated separately in session 11).
                    class_weight="balanced",
                    # lbfgs defaults to 100 iterations and does not converge on this
                    # feature matrix once one-hot expands it.
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
