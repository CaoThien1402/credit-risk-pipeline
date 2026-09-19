"""Mapping SHAP's transformed-space output back to the raw fields the form collects.

Separate from app.py on purpose: app.py is a Streamlit script that loads
models/at_application_model.pkl at import time, so it can't be imported in CI (the
.pkl is gitignored). These functions are pure arithmetic over a SHAP row, so keeping
them here lets tests/test_app_explanations.py cover the form/SHAP seam in CI, where
the model itself is unavailable.
"""


def describe_transformed_feature(transformed_name: str, raw_inputs: dict, categorical_cols: list[str]) -> tuple[str, object]:
    """Map a one-hot-expanded SHAP feature name back to the raw field + the applicant's
    actual (human-readable) value for it - a credit officer reads "home_ownership: RENT",
    not "home_ownership_RENT" or a standardized -0.19 for a numeric column."""
    for col in categorical_cols:
        if transformed_name == col or transformed_name.startswith(col + "_"):
            return col, raw_inputs[col]
    return transformed_name, raw_inputs[transformed_name]


def top_reasons(shap_row, transformed_names: list[str], raw_inputs: dict, categorical_cols: list[str], n: int = 3):
    """Top-n raw features by |SHAP|, summing each categorical's one-hot columns first.

    OneHotEncoder turns one raw field into several transformed columns, and SHAP scores
    each separately. Ranking the transformed columns directly has two failure modes a
    credit officer would see: the same field can occupy two rows of a "top 3 reasons"
    table with the same label and different numbers (e.g. default_on_file_N and
    default_on_file_Y), and a field whose signal is spread across its categories can be
    ranked below a numeric one that is actually less important - the same
    split-importance problem CLAUDE.md cites for not keeping two correlated numerics.
    Summing over a field's own one-hot columns is the right aggregation because SHAP
    values are additive per prediction: the sum is that field's total contribution to
    this decision's log-odds.

    Returns a list of (raw_field, applicant_value, summed_contribution).
    """
    totals: dict[str, float] = {}
    values: dict[str, object] = {}
    for i, tname in enumerate(transformed_names):
        raw_name, raw_value = describe_transformed_feature(tname, raw_inputs, categorical_cols)
        totals[raw_name] = totals.get(raw_name, 0.0) + float(shap_row[i])
        values[raw_name] = raw_value
    ranked = sorted(totals, key=lambda k: -abs(totals[k]))[:n]
    return [(name, values[name], totals[name]) for name in ranked]
