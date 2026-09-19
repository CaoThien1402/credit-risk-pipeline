"""Streamlit app with two tabs: new-application prediction and portfolio dashboard."""
# Two separate model bundles: at_application_model.pkl (Tab 1, no loan_grade/loan_int_rate)
# and portfolio_risk_model.pkl (Tab 2 only) — see the leakage note in sql/schema.sql.
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(REPO_ROOT))

from utils import load_model_bundle
from model.features import split_numeric_categorical
from model.threshold import DECISION_THRESHOLD

st.set_page_config(page_title="Credit Risk & Loan Approval", page_icon="💳", layout="wide")

AT_APPLICATION_MODEL_PATH = REPO_ROOT / "models" / "at_application_model.pkl"

# Real distinct values from ml_features (queried directly, not guessed) — matches training
# data exactly, so OneHotEncoder never has to fall back on handle_unknown="ignore" for a
# value that was actually always valid.
CATEGORICAL_OPTIONS = {
    "loan_intent": ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"],
    "home_ownership": ["MORTGAGE", "OTHER", "OWN", "RENT"],
    "default_on_file": ["N", "Y"],
    "country": ["Canada", "UK", "USA"],
    "education_level": ["Bachelor", "High School", "Master", "PhD"],
    "employment_type": ["Full-time", "Part-time", "Self-employed", "Unemployed"],
}

# (min, max, default, step) — default is the real median from ml_features, bounds are the
# real observed min/max with modest headroom, not arbitrary round numbers.
FLOAT_RANGES = {
    "loan_amnt": (500.0, 40_000.0, 8_000.0, 100.0),
    "loan_percent_income": (0.0, 1.0, 0.15, 0.01),
    "debt_to_income_ratio": (0.0, 1.2, 0.33, 0.01),
    "income": (0.0, 10_000_000.0, 55_000.0, 1_000.0),
    "emp_length": (0.0, 60.0, 4.0, 0.5),
    "credit_utilization_ratio": (0.0, 1.0, 0.5, 0.01),
    "other_debt": (0.0, 2_000_000.0, 9_000.0, 100.0),
}
INT_RANGES = {
    "age": (18, 100, 26, 1),
    "cred_hist_length": (0, 30, 4, 1),
    "past_delinquencies": (0, 20, 0, 1),
    "open_accounts": (0, 20, 8, 1),
}
LOAN_TERM_OPTIONS = [12, 24, 36, 60]  # schema.sql CHECK constraint - only these 4 are valid


@st.cache_resource
def get_at_application_bundle() -> dict:
    return load_model_bundle(str(AT_APPLICATION_MODEL_PATH))


@st.cache_resource
def get_shap_explainer(_model) -> "shap.TreeExplainer":
    # Leading underscore tells st.cache_resource to skip hashing the model - it isn't
    # trivially hashable, and this app only ever loads one at_application model per
    # session, so there's no risk of silently reusing a stale explainer for a different
    # model. Same TreeExplainer setup as notebooks/05_explainability_and_bundles.ipynb.
    return shap.TreeExplainer(_model)


def describe_transformed_feature(transformed_name: str, raw_inputs: dict, categorical_cols: list[str]) -> tuple[str, object]:
    """Map a one-hot-expanded SHAP feature name back to the raw field + the applicant's
    actual (human-readable) value for it - a credit officer reads "home_ownership: RENT",
    not "home_ownership_RENT" or a standardized -0.19 for a numeric column."""
    for col in categorical_cols:
        if transformed_name == col or transformed_name.startswith(col + "_"):
            return col, raw_inputs[col]
    return transformed_name, raw_inputs[transformed_name]


bundle = get_at_application_bundle()

with st.sidebar:
    st.subheader("At-application model")
    _metrics = bundle["metrics"]
    st.metric("ROC-AUC (5-fold CV)", f"{_metrics['roc_auc']:.4f} ± {_metrics['roc_auc_std']:.4f}")
    st.metric("PR-AUC (5-fold CV)", f"{_metrics['pr_auc']:.4f} ± {_metrics['pr_auc_std']:.4f}")
    st.metric("Decision threshold", f"{DECISION_THRESHOLD:.2f}")
    st.caption(f"Trained on {bundle['trained_on_n_rows']:,} historical rows")
    st.caption(f"Trained at: {bundle['trained_at']}")
    st.caption(
        "Excludes loan_grade/loan_int_rate (leakage) and gender/marital_status "
        "(fair lending) — see CLAUDE.md."
    )

tab_predict, tab_dashboard = st.tabs(["New Application Prediction", "Portfolio Dashboard"])

with tab_predict:
    st.header("Approve/Reject New Loan Application")
    st.caption(
        "This model does not use loan_grade/loan_int_rate as input — both are outputs "
        "of underwriting, not data available at application time."
    )

    feature_names = bundle["feature_names"]
    numeric_cols, categorical_cols = split_numeric_categorical(feature_names)
    fields = sorted(feature_names)  # stable, deterministic form layout

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        inputs = {}
        for i, col in enumerate(fields):
            target = col1 if i % 2 == 0 else col2
            if col == "loan_term_months":
                inputs[col] = target.selectbox(col, LOAN_TERM_OPTIONS, index=2)
            elif col in categorical_cols:
                inputs[col] = target.selectbox(col, CATEGORICAL_OPTIONS[col])
            elif col in INT_RANGES:
                lo, hi, default, step = INT_RANGES[col]
                inputs[col] = target.number_input(col, min_value=lo, max_value=hi, value=default, step=step)
            else:
                lo, hi, default, step = FLOAT_RANGES[col]
                inputs[col] = target.number_input(col, min_value=lo, max_value=hi, value=default, step=step)

        submitted = st.form_submit_button("Evaluate application")

    if submitted:
        # Mirrors schema.sql's CHECK (emp_length <= age - 14) — an application violating
        # this couldn't exist in the training data, so the model has never seen anything
        # like it; better to say so than to silently score it anyway.
        max_emp_length = inputs["age"] - 14
        if inputs["emp_length"] > max_emp_length:
            st.error(
                f"emp_length ({inputs['emp_length']:.1f}) can't exceed age - 14 "
                f"({max_emp_length:.0f} for age {inputs['age']:.0f}) — adjust the inputs."
            )
        else:
            input_df = pd.DataFrame([inputs])[feature_names]
            transformed = bundle["preprocessor"].transform(input_df)
            proba_default = float(bundle["model"].predict_proba(transformed)[0, 1])

            reject = proba_default >= DECISION_THRESHOLD
            decision = "REJECT" if reject else "APPROVE"

            result_col, explain_col = st.columns([1, 1])
            with result_col:
                if reject:
                    st.error(f"### Decision: {decision}")
                else:
                    st.success(f"### Decision: {decision}")
                st.metric("Predicted P(default)", f"{proba_default:.1%}")
                st.caption(f"Decision threshold: {DECISION_THRESHOLD:.2f} (reject if P(default) ≥ threshold)")

            with explain_col:
                with st.expander("Why this threshold, not 0.5?"):
                    st.markdown(
                        "This model's `predict_proba` is not a calibrated probability — "
                        "`scale_pos_weight` (used to correct for the ~21.8% default base "
                        "rate during training) shifts its scores away from true "
                        "probabilities. The threshold above was instead found by sweeping "
                        "every cutoff against a held-out test set and picking the one that "
                        "minimizes real expected cost (loan principal lost on a missed "
                        "default vs. interest income lost on a wrongly rejected good "
                        "applicant). Full derivation: "
                        "`notebooks/06_decision_threshold.ipynb`."
                    )

            st.divider()
            st.subheader("Why this prediction?")

            transformed_names = list(bundle["preprocessor"].get_feature_names_out())
            row_df = pd.DataFrame(transformed, columns=transformed_names)

            explainer = get_shap_explainer(bundle["model"])
            row_shap_values = explainer.shap_values(row_df)[0]

            top_3_idx = np.argsort(-np.abs(row_shap_values))[:3]
            top_3 = pd.DataFrame(
                [
                    {
                        "Feature": describe_transformed_feature(transformed_names[i], inputs, categorical_cols)[0],
                        "Applicant's value": describe_transformed_feature(transformed_names[i], inputs, categorical_cols)[1],
                        "Effect": "increases risk" if row_shap_values[i] > 0 else "decreases risk",
                        "Contribution": round(float(row_shap_values[i]), 4),
                    }
                    for i in top_3_idx
                ]
            )

            # Stacked full-width, not side by side: in a narrow column the table's
            # Effect/Contribution columns (the ones that actually answer "why") scroll
            # out of view, and the force plot's feature labels overlap into noise.
            st.markdown("**Top 3 reasons for this decision** (largest |SHAP| impact):")
            # st.table, not st.dataframe: 3 static rows need no sorting/scrolling, and
            # st.dataframe draws to a canvas - its toolbar floated over the force plot
            # below, and its cell text isn't in the DOM, so nothing could assert on it.
            st.table(top_3.set_index("Feature"))

            # Rounded so labels read "income = -0.18", not "-0.1786779784755...".
            # contribution_threshold hides labels on small segments. With 35 transformed
            # features most contribute almost nothing, and labelling all of them made the
            # text collide (loan_intent_VENTURE over loan_intent_DEBTCONSOLIDATION, etc).
            # The top-3 table above already names the drivers, so nothing is lost.
            fig = shap.force_plot(
                explainer.expected_value, row_shap_values, row_df.iloc[0].round(2),
                matplotlib=True, show=False, figsize=(16, 3.2),
                contribution_threshold=0.12,
            )
            st.pyplot(fig, clear_figure=True)
            st.caption(
                "Red segments push the prediction toward higher risk (reject); blue "
                "segments push it toward lower risk (approve). The plot is in the "
                "model's transformed feature space (scaled numerics, one-hot "
                "categoricals), so its values won't match what you typed - the table "
                "above translates the same features back. Its axis is log-odds, not "
                "the probability shown at the top, so f(x) here is not comparable to "
                "P(default) directly."
            )

with tab_dashboard:
    st.header("Portfolio Management Dashboard")
    # TODO: Plotly charts reading directly from Postgres (WHERE data_source='historical'
    # for historical figures; can union synthetic_daily to track the live stream, but
    # label the two sources separately in the chart).
    # TODO: scatter_geo via cities.latitude/longitude (joined through customers.city_id).
    st.info("TODO: monthly outstanding balance chart, default rate by grade/country, scatter_geo map.")
