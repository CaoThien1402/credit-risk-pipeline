"""Streamlit app with two tabs: new-application prediction and portfolio dashboard."""
# Two separate model bundles: at_application_model.pkl (Tab 1, no loan_grade/loan_int_rate)
# and portfolio_risk_model.pkl (Tab 2 only) — see the leakage note in sql/schema.sql.
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(REPO_ROOT))

from utils import load_model_bundle
from model.features import split_numeric_categorical
from model.threshold import DECISION_THRESHOLD

st.set_page_config(page_title="Credit Risk & Loan Approval", layout="wide")

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


tab_predict, tab_dashboard = st.tabs(["New Application Prediction", "Portfolio Dashboard"])

with tab_predict:
    st.header("Approve/Reject New Loan Application")
    st.caption(
        "This model does not use loan_grade/loan_int_rate as input — both are outputs "
        "of underwriting, not data available at application time."
    )

    bundle = get_at_application_bundle()
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
                st.info("SHAP force plot with top-3 reasons — buổi 14, not yet built.")

with tab_dashboard:
    st.header("Portfolio Management Dashboard")
    # TODO: Plotly charts reading directly from Postgres (WHERE data_source='historical'
    # for historical figures; can union synthetic_daily to track the live stream, but
    # label the two sources separately in the chart).
    # TODO: scatter_geo via cities.latitude/longitude (joined through customers.city_id).
    st.info("TODO: monthly outstanding balance chart, default rate by grade/country, scatter_geo map.")
