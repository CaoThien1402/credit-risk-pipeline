"""Unit tests for the ETL helper functions."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "etl"))
import pandas as pd
from historical_load import clean_outliers, drop_redundant_columns


def test_clean_outliers_caps_impossible_age():
    df = pd.DataFrame({"person_age": [25, 144], "person_emp_length": [3, 4]})
    out = clean_outliers(df)
    assert out["person_age"].isna().sum() == 1


def test_drop_redundant_columns():
    df = pd.DataFrame({"loan_percent_income": [0.1], "loan_to_income_ratio": [0.1]})
    out = drop_redundant_columns(df)
    assert "loan_to_income_ratio" not in out.columns
