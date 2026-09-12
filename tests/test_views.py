"""Structural tests for sql/schema.sql and sql/views.sql.

Needs a reachable Postgres with both files already applied - true in CI (see
.github/workflows/ci.yml, which spins up a postgres service and applies schema.sql +
views.sql before running tests) and true locally whenever `docker compose up -d` has
run. Skips (doesn't fail) if no DB is reachable, so `uv run pytest` still works without
Docker running.

Deliberately structure-only, not row-count-based: CI's database is schema-only
(nothing inserted), while a local dev database is fully seeded. Column sets and
constraints are true in both.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "etl"))

import pytest
from sqlalchemy import inspect, text

from config import get_engine

EXPECTED_TABLES = {"cities", "customers", "credit_bureau", "loans"}

# sql/views.sql's ml_features SELECT list. Kept here (not queried from information_schema
# in a loop) so a diff against this set is readable in a failure message.
ML_FEATURES_COLUMNS = {
    "loan_id", "client_id", "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate",
    "loan_percent_income", "debt_to_income_ratio", "loan_status", "age", "income",
    "home_ownership", "emp_length", "default_on_file", "cred_hist_length",
    "credit_utilization_ratio", "past_delinquencies", "country", "loan_term_months",
    "other_debt", "gender", "marital_status", "education_level", "employment_type",
    "open_accounts",
}

DASHBOARD_WINDOW_COLUMNS = {
    "avg_loan_amnt_by_age_bucket", "default_rate_by_grade", "util_rank_in_country",
}


@pytest.fixture(scope="module")
def engine():
    eng = get_engine()
    try:
        with eng.connect():
            pass
    except Exception as exc:
        pytest.skip(f"no reachable Postgres ({exc}) - run `docker compose up -d` first")
    return eng


def _columns(engine, table_name):
    return {c["name"] for c in inspect(engine).get_columns(table_name)}


def test_schema_has_exactly_four_tables(engine):
    """Guards the 'so co 4 bang, khong phai 5' mistake CLAUDE.md warns has recurred twice."""
    tables = set(inspect(engine).get_table_names())
    assert tables == EXPECTED_TABLES


def test_ml_features_has_exactly_the_documented_columns(engine):
    assert _columns(engine, "ml_features") == ML_FEATURES_COLUMNS


def test_ml_features_excludes_the_dashboard_window_columns(engine):
    """The structural safeguard CLAUDE.md describes: training code queries ml_features,
    which must not expose default_rate_by_grade (= AVG(loan_status), the target itself)
    or the other two population-level window columns."""
    assert not (_columns(engine, "ml_features") & DASHBOARD_WINDOW_COLUMNS)


def test_dashboard_aggregates_adds_exactly_the_three_window_columns(engine):
    dashboard_cols = _columns(engine, "dashboard_aggregates")
    assert dashboard_cols == ML_FEATURES_COLUMNS | DASHBOARD_WINDOW_COLUMNS


def test_application_ref_is_the_unique_constraint_not_loan_id(engine):
    """loan_id is SERIAL; ON CONFLICT (loan_id) would never fire. application_ref must
    carry the real UNIQUE constraint that daily_ingest.py's UPSERT depends on."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT pg_get_constraintdef(con.oid) AS definition
            FROM pg_constraint con
            JOIN pg_class rel ON rel.oid = con.conrelid
            WHERE rel.relname = 'loans' AND con.contype IN ('u', 'p')
        """)).fetchall()
    definitions = {row.definition for row in rows}
    assert "UNIQUE (application_ref)" in definitions
    assert "PRIMARY KEY (loan_id)" in definitions
