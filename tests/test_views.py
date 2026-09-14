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

# etl/ for config, repo root for model/ - both needed. The repo-root entry is what makes
# `from model.features import ...` below resolve; without it this module fails at import
# time, which is a collection error, not a skip (the no-Postgres skip lives in the engine
# fixture and never gets a chance to run).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "etl"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from sqlalchemy import inspect, text

from config import get_engine
from model.features import ML_FEATURES_COLUMNS

EXPECTED_TABLES = {"cities", "customers", "credit_bureau", "loans"}

# model.features.ML_FEATURES_COLUMNS is a tuple in view order; these tests compare against
# column sets from information_schema, so set() it here. Single source of truth - this
# list used to be duplicated between here and test_features.py with nothing enforcing
# that the two copies matched.
EXPECTED_ML_FEATURES_COLUMNS = set(ML_FEATURES_COLUMNS)

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
    assert _columns(engine, "ml_features") == EXPECTED_ML_FEATURES_COLUMNS


def test_ml_features_excludes_the_dashboard_window_columns(engine):
    """The structural safeguard CLAUDE.md describes: training code queries ml_features,
    which must not expose default_rate_by_grade (= AVG(loan_status), the target itself)
    or the other two population-level window columns."""
    assert not (_columns(engine, "ml_features") & DASHBOARD_WINDOW_COLUMNS)


def test_dashboard_aggregates_adds_exactly_the_three_window_columns(engine):
    dashboard_cols = _columns(engine, "dashboard_aggregates")
    assert dashboard_cols == EXPECTED_ML_FEATURES_COLUMNS | DASHBOARD_WINDOW_COLUMNS


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
