"""Column bookkeeping for ml_features, shared by sessions 9-12.

Decisions here (which columns are leakage, which are identifiers) come from buổi 8's
EDA (notebooks/01_eda.ipynb) and are documented in CLAUDE.md / docs/MEMORY.md. Nothing
in this file is new analysis - it's just applying that already-made decision.
"""

ID_COLS = ["loan_id", "client_id"]
TARGET_COL = "loan_status"

# Near-deterministic with the target (buổi 8) - excluded from the at-application
# feature set, kept only in the separate portfolio-risk model. See CLAUDE.md.
LEAKAGE_COLS = ["loan_grade", "loan_int_rate"]

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
    """Feature columns for 'portfolio' (all) or 'at_application' (leakage cols dropped)."""
    if feature_set not in ("portfolio", "at_application"):
        raise ValueError(f"feature_set must be 'portfolio' or 'at_application', got {feature_set!r}")
    cols = [c for c in columns if c not in ID_COLS and c != TARGET_COL]
    if feature_set == "at_application":
        cols = [c for c in cols if c not in LEAKAGE_COLS]
    return cols


def split_numeric_categorical(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Partition feature_cols into (numeric_cols, categorical_cols) for the ColumnTransformer."""
    known_categorical = NOMINAL_CATEGORICAL_COLS + ORDINAL_CATEGORICAL_COLS
    categorical_cols = [c for c in feature_cols if c in known_categorical]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    return numeric_cols, categorical_cols
