"""Unit tests for etl/historical_load.py's pure transformation functions.

Covers clean_outliers, impute_missing, check_schema_invariants,
drop_redundant_columns, backfill_loan_dates and split_into_tables - everything in the
module that transforms a dataframe without touching the database. load_to_postgres is
deliberately not tested here: it needs a live connection, which would turn this file
into an integration test. The schema it writes into is already covered structurally by
tests/test_views.py.
"""
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "etl"))

import pandas as pd
import pytest

from historical_load import (
    backfill_loan_dates,
    check_schema_invariants,
    clean_outliers,
    drop_redundant_columns,
    impute_missing,
    split_into_tables,
)


def test_clean_outliers_caps_impossible_age():
    df = pd.DataFrame({"person_age": [25, 144], "person_emp_length": [3, 4]})
    out = clean_outliers(df)
    assert out["person_age"].isna().sum() == 1


def test_clean_outliers_flags_emp_length_inconsistent_with_age():
    df = pd.DataFrame({"person_age": [21, 40], "person_emp_length": [123, 5]})
    out = clean_outliers(df)
    assert pd.isna(out.loc[0, "person_emp_length"])
    assert out.loc[1, "person_emp_length"] == 5


def test_clean_outliers_does_not_let_emp_length_ride_out_on_an_invalid_age():
    """The latent bug this guards: if age is nulled *before* emp_length is compared
    against it, the comparison is against NaN, which is False for every row - so an
    unvalidated emp_length survives. impute_missing then fills age with the median
    (well under 144), and the row violates schema.sql's CHECK (emp_length <= age - 14)
    at insert time. No row in the real data hits both conditions today, so only a
    constructed row can catch a regression here.
    """
    df = pd.DataFrame({"person_age": [144], "person_emp_length": [123]})

    out = clean_outliers(df)

    assert pd.isna(out.loc[0, "person_age"])
    assert pd.isna(out.loc[0, "person_emp_length"]), (
        "emp_length was kept although the age it should have been validated against "
        "was rejected - it will violate CHECK (emp_length <= age - 14) after imputation"
    )


# --- impute_missing ----------------------------------------------------------------

def test_impute_missing_leaves_no_null_age():
    """customers.age is NOT NULL (schema.sql), but clean_outliers only caps to NaN.
    CLAUDE.md flags this line as load-bearing: without it the historical load fails on
    the NOT NULL constraint."""
    df = pd.DataFrame({
        "person_age": [25.0, None, 35.0],
        "person_emp_length": [3.0, 4.0, None],
        "loan_int_rate": [10.0, None, 12.0],
    })

    out = impute_missing(df)

    assert out["person_age"].notna().all()
    assert out["person_emp_length"].notna().all()
    assert out["loan_int_rate"].notna().all()


def test_impute_missing_uses_the_median_of_the_surviving_values():
    df = pd.DataFrame({
        "person_age": [20.0, 30.0, 40.0, None],
        "person_emp_length": [1.0, 2.0, 3.0, None],
        "loan_int_rate": [10.0, 20.0, 30.0, None],
    })

    out = impute_missing(df)

    assert out.loc[3, "person_age"] == 30.0
    assert out.loc[3, "person_emp_length"] == 2.0
    assert out.loc[3, "loan_int_rate"] == 20.0


def test_capping_before_imputing_keeps_outliers_out_of_the_median():
    """CLAUDE.md's 'cap outliers before imputing, not after' constraint, as behaviour
    rather than prose. Run in the wrong order the 144s are still present when the
    median is taken and drag it upward; the ordering in __main__ is what keeps the
    imputed value equal to the median of the plausible ages alone.
    """
    df = pd.DataFrame({
        "person_age": [20.0, 22.0, 24.0, 144.0, 144.0, None],
        "person_emp_length": [1.0, 2.0, 3.0, 4.0, 5.0, None],
        "loan_int_rate": [10.0] * 6,
    })

    right_way = impute_missing(clean_outliers(df))
    wrong_way = impute_missing(df)

    assert right_way.loc[5, "person_age"] == 22.0, "median should ignore the 144s"
    assert wrong_way.loc[5, "person_age"] == 24.0
    assert right_way.loc[5, "person_age"] != wrong_way.loc[5, "person_age"], (
        "if these ever match, this test has stopped proving the ordering matters"
    )


# --- check_schema_invariants -------------------------------------------------------

def test_check_schema_invariants_passes_the_real_pipeline_order():
    df = pd.DataFrame({
        "person_age": [20.0, 22.0, 144.0, None],
        "person_emp_length": [1.0, 2.0, 123.0, None],
        "loan_int_rate": [10.0] * 4,
    })

    cleaned = impute_missing(clean_outliers(df))

    assert check_schema_invariants(cleaned) is cleaned


def test_check_schema_invariants_rejects_a_null_age():
    df = pd.DataFrame({"person_age": [25.0, None], "person_emp_length": [1.0, 2.0]})
    with pytest.raises(ValueError, match="null person_age"):
        check_schema_invariants(df)


def test_check_schema_invariants_rejects_emp_length_above_age_minus_14():
    df = pd.DataFrame({"person_age": [26.0], "person_emp_length": [20.0]})
    with pytest.raises(ValueError, match="emp_length <= age - 14"):
        check_schema_invariants(df)


def test_drop_redundant_columns():
    df = pd.DataFrame({"loan_percent_income": [0.1], "loan_to_income_ratio": [0.1]})
    out = drop_redundant_columns(df)
    assert "loan_to_income_ratio" not in out.columns


# --- backfill_loan_dates -----------------------------------------------------------

def _rows(n=25):
    """backfill_loan_dates only reads len(df), so the contents don't matter."""
    return pd.DataFrame({"client_ID": [f"CUST_{i:05d}" for i in range(n)]})


def test_backfill_loan_dates_adds_the_three_synthetic_columns():
    out = backfill_loan_dates(_rows())
    assert {"loan_date", "data_source", "application_ref"} <= set(out.columns)
    assert (out["data_source"] == "historical").all()


def test_backfill_loan_dates_stays_inside_the_requested_window():
    """The source data has no date column at all - these are invented, so they must at
    least land in the window the caller asked for."""
    months_back = 24
    out = backfill_loan_dates(_rows(), months_back=months_back)

    end = datetime.today()
    start = end - timedelta(days=months_back * 30)
    assert out["loan_date"].min() >= start
    assert out["loan_date"].max() <= end


def test_backfill_loan_dates_seed_reproduces_offsets_but_not_refs_or_exact_instants():
    """Two things the seed does NOT control, both found by this test failing first:

    1. application_ref comes from uuid4(), which numpy's Generator doesn't seed - a
       same-seed rerun produces entirely different refs.
    2. The date window is anchored on datetime.today() *at call time*, so two runs
       milliseconds apart get different anchors and the absolute timestamps drift by
       that much (observed: ~1.2 microseconds between back-to-back calls). The seeded
       offsets are identical; the origin they're added to is not. So this function is
       reproducible in shape, not bit-for-bit - asserting exact equality here passes on
       a fast machine and fails on a slow one, which is worse than not testing it.
    """
    a = backfill_loan_dates(_rows(), seed=7)
    b = backfill_loan_dates(_rows(), seed=7)

    drift = (a["loan_date"] - b["loan_date"]).abs().max()
    assert drift < timedelta(seconds=1), f"same-seed dates drifted by {drift}"

    spacing_a = a["loan_date"].sort_values().diff().dropna().reset_index(drop=True)
    spacing_b = b["loan_date"].sort_values().diff().dropna().reset_index(drop=True)
    pd.testing.assert_series_equal(spacing_a, spacing_b)

    assert set(a["application_ref"]).isdisjoint(set(b["application_ref"]))


def test_backfill_loan_dates_application_refs_are_unique():
    """application_ref is the UPSERT key (UNIQUE in schema.sql); a duplicate inside one
    load would fail the insert."""
    out = backfill_loan_dates(_rows(50))
    assert out["application_ref"].is_unique


# --- split_into_tables -------------------------------------------------------------

def _flat_source(n_cities=2):
    """One row per loan, in the shape historical_load sees after the earlier steps."""
    cities = [("USA", "CA", "Los Angeles", 34.05, -118.24),
              ("UK", "ENG", "London", 51.51, -0.13)]
    rows = []
    for i in range(4):
        country, state, city, lat, lon = cities[i % n_cities]
        rows.append({
            "client_ID": f"CUST_{i:05d}",
            "person_age": 30 + i, "person_income": 50_000 + i,
            "person_home_ownership": "RENT", "person_emp_length": 5,
            "gender": "Female", "marital_status": "Single",
            "education_level": "Bachelor", "employment_type": "Full-time",
            "country": country, "state": state, "city": city,
            "city_latitude": lat, "city_longitude": lon,
            "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 4,
            "open_accounts": 6, "credit_utilization_ratio": 0.3,
            "past_delinquencies": 0,
            "application_ref": f"ref-{i}", "loan_intent": "PERSONAL",
            "loan_grade": "B", "loan_amnt": 10_000.0, "loan_int_rate": 11.5,
            "loan_term_months": 36, "loan_status": 0, "loan_percent_income": 0.2,
            "other_debt": 1_000.0, "debt_to_income_ratio": 0.25,
            "loan_date": datetime(2026, 1, 1), "data_source": "historical",
        })
    return pd.DataFrame(rows)


def test_split_into_tables_returns_exactly_the_four_schema_tables():
    """CLAUDE.md flags that this has been mis-stated as 5 tables twice."""
    tables = split_into_tables(_flat_source())
    assert set(tables) == {"cities", "customers", "credit_bureau", "loans"}


def test_split_into_tables_deduplicates_cities():
    """18 distinct city combos across 32,581 rows is the whole reason cities is its own
    dimension table - one row per combo, not per loan."""
    df = _flat_source(n_cities=2)
    tables = split_into_tables(df)
    assert len(tables["cities"]) == 2
    assert len(df) == 4


def test_split_into_tables_renames_source_columns_to_schema_names():
    customers = split_into_tables(_flat_source())["customers"]
    assert {"client_id", "age", "income", "home_ownership", "emp_length"} <= set(customers.columns)
    assert not {"client_ID", "person_age", "person_income"} & set(customers.columns)


def test_split_into_tables_keeps_application_ref_on_loans():
    """application_ref, not loan_id, is the UPSERT key - if the split dropped it the
    daily ingest's ON CONFLICT would have nothing to match on."""
    loans = split_into_tables(_flat_source())["loans"]
    assert "application_ref" in loans.columns
    assert "loan_id" not in loans.columns


def test_split_into_tables_carries_city_columns_for_later_fk_resolution():
    """customers keeps country/state/city until load_to_postgres swaps them for the
    SERIAL city_id, which only exists after cities is inserted."""
    customers = split_into_tables(_flat_source())["customers"]
    assert {"country", "state", "city"} <= set(customers.columns)
