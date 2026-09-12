"""Tests for etl/daily_ingest.py's sampling logic (pure function, no DB needed)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "etl"))

import numpy as np
import pandas as pd
import pytest

from daily_ingest import CATEGORICAL_COLS, NUMERIC_COLS, build_daily_batch, sample_new_applications


def _reference_stats(n=20):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "client_id": [f"c{i}" for i in range(n)],
        "loan_amnt": rng.uniform(1000, 20000, n),
        "loan_int_rate": rng.uniform(5, 20, n),
        "loan_percent_income": rng.uniform(0.05, 0.5, n),
        "other_debt": rng.uniform(0, 5000, n),
        "debt_to_income_ratio": rng.uniform(0.05, 0.4, n),
        "loan_intent": rng.choice(["EDUCATION", "MEDICAL", "VENTURE"], n),
        "loan_grade": rng.choice(["A", "B", "C", "D"], n),
        "loan_term_months": rng.choice([12, 24, 36, 60], n),
    })


def test_same_seed_reproduces_the_same_batch():
    """The failure this guards: a dead seed parameter that looks reproducible but isn't."""
    ref = _reference_stats()
    a = sample_new_applications(10, ref, seed=42)
    b = sample_new_applications(10, ref, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_gives_a_different_batch():
    ref = _reference_stats()
    a = sample_new_applications(10, ref, seed=42)
    b = sample_new_applications(10, ref, seed=43)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(a, b)


def test_no_seed_still_produces_a_valid_batch():
    """seed=None (the default) must still work - only reproducibility is opt-in."""
    ref = _reference_stats()
    out = sample_new_applications(10, ref, seed=None)
    assert len(out) == 10
    assert set(out["client_id"]).issubset(set(ref["client_id"]))


def test_numeric_columns_respect_their_floor():
    """NUMERIC_MIN clipping (e.g. loan_amnt > 0) must hold even with Gaussian noise."""
    ref = _reference_stats()
    out = sample_new_applications(200, ref, seed=1)
    for col in NUMERIC_COLS:
        assert (out[col] >= 0).all(), f"{col} went negative"


def test_categorical_columns_only_draw_seen_values():
    ref = _reference_stats()
    out = sample_new_applications(200, ref, seed=1)
    for col in CATEGORICAL_COLS:
        assert set(out[col].unique()).issubset(set(ref[col].unique()))


def test_build_daily_batch_is_fully_reproducible_including_n():
    """Regression guard for the __main__ bug this module used to have: n was drawn from
    legacy np.random.randint, uninfluenced by any seed. sample_new_applications itself
    was always correctly seeded, so a naive check ("does this seed reconstruct a batch
    of the same size n?") passes even on the buggy code - it doesn't test whether *n*
    is reproducible, only that sampling is once n is already known.

    The real contract: calling build_daily_batch twice with the same seed must yield
    the same n AND the same rows, with no other input in between. Verified this fails
    on a reconstruction of the old n = int(np.random.randint(50, 101)) logic (different
    n almost every call, same seed) before writing the fix.
    """
    ref = _reference_stats()
    seed_a, batch_a = build_daily_batch(ref, seed=123)
    seed_b, batch_b = build_daily_batch(ref, seed=123)
    assert seed_a == seed_b == 123
    pd.testing.assert_frame_equal(batch_a, batch_b)


def test_build_daily_batch_generates_a_seed_when_none_given():
    ref = _reference_stats()
    seed, batch = build_daily_batch(ref, seed=None)
    assert isinstance(seed, int)
    assert len(batch) >= 50 and len(batch) <= 100
