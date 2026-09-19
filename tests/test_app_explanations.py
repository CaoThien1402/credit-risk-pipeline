"""Unit tests for app.py's SHAP-to-form mapping (pure functions, no model or DB needed).

These guard the seam where a wiring bug is least visible: SHAP scores the
one-hot-expanded transformed columns, but the form - and the officer reading the
result - works in raw fields. Everything below is arithmetic on a fake SHAP row, so it
runs in CI where models/*.pkl doesn't exist.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from explanations import describe_transformed_feature, top_reasons

CATEGORICAL = ["home_ownership", "default_on_file"]
INPUTS = {
    "home_ownership": "RENT",
    "default_on_file": "Y",
    "income": 85000.0,
    "loan_amnt": 25000.0,
}


def test_one_hot_name_maps_back_to_raw_field_and_the_applicants_own_value():
    # The column that fired is default_on_file_N, but this applicant is 'Y'. Reporting
    # "default_on_file = N" here would tell the officer the opposite of the truth.
    assert describe_transformed_feature("default_on_file_N", INPUTS, CATEGORICAL) == ("default_on_file", "Y")
    assert describe_transformed_feature("home_ownership_MORTGAGE", INPUTS, CATEGORICAL) == ("home_ownership", "RENT")


def test_numeric_name_passes_through_with_the_raw_not_standardized_value():
    assert describe_transformed_feature("income", INPUTS, CATEGORICAL) == ("income", 85000.0)


def test_one_hot_columns_of_the_same_field_are_summed_not_listed_twice():
    # Without aggregation this yields two rows both labelled "default_on_file = Y"
    # (+0.6 and +0.3) and pushes income out of the top 3 despite being larger in total.
    names = ["default_on_file_N", "default_on_file_Y", "income", "loan_amnt"]
    shap_row = [0.6, 0.3, -0.8, 0.1]

    result = top_reasons(shap_row, names, INPUTS, CATEGORICAL, n=3)
    fields = [r[0] for r in result]

    assert fields == ["default_on_file", "income", "loan_amnt"]
    assert len(set(fields)) == len(fields), "a raw field must not occupy two rows"
    assert result[0][2] == pytest.approx(0.9)
    assert result[1][2] == pytest.approx(-0.8)


def test_split_categorical_outranks_a_numeric_only_when_its_total_is_larger():
    # Each one-hot column alone (0.3/0.3/0.3) is smaller than income's 0.7, but the
    # field's real contribution to this prediction is 0.9 - it should rank first.
    names = ["home_ownership_RENT", "home_ownership_OWN", "home_ownership_MORTGAGE", "income"]
    shap_row = [0.3, 0.3, 0.3, -0.7]

    result = top_reasons(shap_row, names, INPUTS, CATEGORICAL, n=2)

    assert [r[0] for r in result] == ["home_ownership", "income"]


def test_opposing_one_hot_contributions_cancel_rather_than_double_count():
    names = ["default_on_file_N", "default_on_file_Y", "income"]
    shap_row = [0.5, -0.5, 0.2]

    result = top_reasons(shap_row, names, INPUTS, CATEGORICAL, n=2)

    assert result[0][0] == "income"
    assert dict((r[0], r[2]) for r in result)["default_on_file"] == pytest.approx(0.0)
