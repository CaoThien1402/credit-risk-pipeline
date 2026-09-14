"""Tests for model/features.py column bookkeeping.

These guard the two ways the feature split silently goes wrong: a string column routed
to StandardScaler, and the two feature sets drifting apart by something other than the
documented leakage columns.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from model.features import (
    ID_COLS,
    LEAKAGE_COLS,
    ML_FEATURES_COLUMNS,
    NOMINAL_CATEGORICAL_COLS,
    ORDINAL_CATEGORICAL_COLS,
    PROTECTED_ATTRIBUTE_COLS,
    TARGET_COL,
    get_feature_columns,
    split_numeric_categorical,
)

# Columns the database stores as text/char. StandardScaler raises
# "could not convert string to float" on any of these.
STRING_COLUMNS = set(NOMINAL_CATEGORICAL_COLS) | set(ORDINAL_CATEGORICAL_COLS)


@pytest.mark.parametrize("feature_set", ["portfolio", "at_application"])
def test_no_string_column_is_classified_as_numeric(feature_set):
    """loan_grade is CHAR(1) but isn't nominal, so it used to fall through to numeric_cols
    and crash StandardScaler in the portfolio pipeline."""
    cols = get_feature_columns(ML_FEATURES_COLUMNS, feature_set)
    numeric_cols, _ = split_numeric_categorical(cols)
    leaked_strings = STRING_COLUMNS & set(numeric_cols)
    assert not leaked_strings, f"string columns routed to StandardScaler: {sorted(leaked_strings)}"


def test_feature_sets_differ_exactly_by_leakage_cols():
    portfolio = set(get_feature_columns(ML_FEATURES_COLUMNS, "portfolio"))
    at_application = set(get_feature_columns(ML_FEATURES_COLUMNS, "at_application"))
    assert portfolio - at_application == set(LEAKAGE_COLS)
    assert at_application - portfolio == set()


@pytest.mark.parametrize("feature_set", ["portfolio", "at_application"])
def test_ids_and_target_never_become_features(feature_set):
    cols = get_feature_columns(ML_FEATURES_COLUMNS, feature_set)
    assert TARGET_COL not in cols
    assert not set(ID_COLS) & set(cols)


@pytest.mark.parametrize("feature_set", ["portfolio", "at_application"])
def test_protected_attributes_excluded_from_both_feature_sets(feature_set):
    """Fair lending (ECOA / Reg B and equivalents): gender and marital_status must not
    reach any model, including the portfolio one. Session 8's scan showed both are noise
    against the ~21.8% base rate, so this costs no accuracy."""
    cols = get_feature_columns(ML_FEATURES_COLUMNS, feature_set)
    leaked = set(PROTECTED_ATTRIBUTE_COLS) & set(cols)
    assert not leaked, f"protected attributes reached the {feature_set} feature set: {sorted(leaked)}"


def test_every_feature_column_is_classified():
    """No column may be silently dropped between get_feature_columns and the split."""
    cols = get_feature_columns(ML_FEATURES_COLUMNS, "portfolio")
    numeric_cols, categorical_cols = split_numeric_categorical(cols)
    assert sorted(numeric_cols + categorical_cols) == sorted(cols)


def test_unknown_feature_set_rejected():
    with pytest.raises(ValueError):
        get_feature_columns(ML_FEATURES_COLUMNS, "everything")
