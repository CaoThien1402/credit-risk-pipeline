"""Sanity guard for model/threshold.py.

The real validation is notebooks/06_decision_threshold.ipynb (executed against the
held-out test set); this just catches an accidental edit turning the constant into
something structurally nonsensical (e.g. a stray edit leaving 5.3 or -0.53).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.threshold import DECISION_THRESHOLD


def test_decision_threshold_is_a_valid_probability_cutoff():
    assert isinstance(DECISION_THRESHOLD, float)
    assert 0.0 < DECISION_THRESHOLD < 1.0
