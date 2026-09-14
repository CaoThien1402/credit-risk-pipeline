"""Column bookkeeping for ml_features, shared by sessions 9-12.

Decisions here (which columns are leakage, which are identifiers) come from buổi 8's
EDA (notebooks/01_eda.ipynb) and are documented in CLAUDE.md / docs/MEMORY.md. Nothing
in this file is new analysis - it's just applying that already-made decision.
"""

# The ml_features view's SELECT list (sql/views.sql), in view order. Single source of
# truth: tests/test_features.py and tests/test_views.py both import this rather than
# keeping their own copies, which previously drifted apart with nothing enforcing a
# match. test_views.py asserts this equals the live view's columns, so an edit to
# sql/views.sql that isn't reflected here fails CI rather than passing quietly.
ML_FEATURES_COLUMNS = (
    "loan_id", "client_id", "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate",
    "loan_percent_income", "debt_to_income_ratio", "loan_status", "age", "income",
    "home_ownership", "emp_length", "default_on_file", "cred_hist_length",
    "credit_utilization_ratio", "past_delinquencies", "country", "loan_term_months",
    "other_debt", "gender", "marital_status", "education_level", "employment_type",
    "open_accounts",
)

ID_COLS = ["loan_id", "client_id"]
TARGET_COL = "loan_status"

# Near-deterministic with the target (buổi 8) - excluded from the at-application
# feature set, kept only in the separate portfolio-risk model. See CLAUDE.md.
LEAKAGE_COLS = ["loan_grade", "loan_int_rate"]

# Protected attributes under consumer-credit fair-lending law (US ECOA / Regulation B,
# with equivalents in Canada and the UK - the three countries in this dataset). Using
# them as model input is a compliance problem independent of how well they predict.
# Excluded from BOTH feature sets, not just at_application: fair-lending exposure doesn't
# disappear because a model analyses an existing book instead of approving a new loan.
# Session 8's scan makes this free - both are statistically noise against the ~21.8% base
# rate (gender 0.11pp default-rate spread, marital_status 0.59pp), so dropping them costs
# no accuracy. Numbers in docs/MEMORY.md.
PROTECTED_ATTRIBUTE_COLS = ["gender", "marital_status"]

NOMINAL_CATEGORICAL_COLS = [
    "home_ownership",
    "loan_intent",
    "gender",
    "marital_status",
    "education_level",
    "employment_type",
    "default_on_file",
    "country",
]

# loan_grade is CHAR(1) ('A'..'G') - a category, not a number. It only reaches a feature
# set in the portfolio case (at_application drops it as leakage), but without listing it
# here it falls through to numeric_cols and StandardScaler raises
# "could not convert string to float: 'A'". Kept separate from NOMINAL_CATEGORICAL_COLS
# because grade is genuinely ordinal (A < B < ... < G); one-hot is the safe default,
# switching it to an OrdinalEncoder is a modelling decision for session 10-11.
ORDINAL_CATEGORICAL_COLS = ["loan_grade"]


def get_feature_columns(columns: list[str], feature_set: str) -> list[str]:
    """Feature columns for 'portfolio' (keeps leakage cols) or 'at_application' (drops
    them). Protected attributes are dropped from both."""
    if feature_set not in ("portfolio", "at_application"):
        raise ValueError(f"feature_set must be 'portfolio' or 'at_application', got {feature_set!r}")
    excluded = set(ID_COLS) | {TARGET_COL} | set(PROTECTED_ATTRIBUTE_COLS)
    if feature_set == "at_application":
        excluded |= set(LEAKAGE_COLS)
    return [c for c in columns if c not in excluded]


def split_numeric_categorical(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Partition feature_cols into (numeric_cols, categorical_cols) for the ColumnTransformer."""
    known_categorical = NOMINAL_CATEGORICAL_COLS + ORDINAL_CATEGORICAL_COLS
    categorical_cols = [c for c in feature_cols if c in known_categorical]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    return numeric_cols, categorical_cols
