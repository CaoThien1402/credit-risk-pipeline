"""Streamlit app with two tabs: new-application prediction and portfolio dashboard."""
# Two separate model bundles: at_application_model.pkl (Tab 1, no loan_grade/loan_int_rate)
# and portfolio_risk_model.pkl (Tab 2 only) — see the leakage note in sql/schema.sql.
import streamlit as st

st.set_page_config(page_title="Credit Risk & Loan Approval", layout="wide")

tab_predict, tab_dashboard = st.tabs(["New Application Prediction", "Portfolio Dashboard"])

with tab_predict:
    st.header("Approve/Reject New Loan Application")
    st.caption(
        "This model does not use loan_grade/loan_int_rate as input — both are outputs "
        "of underwriting, not data available at application time."
    )
    # TODO: input form (age, income, loan amount, loan intent, home_ownership, emp_length,
    # credit_bureau fields) — do not add loan_grade/loan_int_rate.
    # TODO: load_model("models/at_application_model.pkl")
    # TODO: SHAP force plot per prediction, showing the top 3 reasons.
    st.info("TODO: input form + load model + SHAP force plot.")

with tab_dashboard:
    st.header("Portfolio Management Dashboard")
    # TODO: Plotly charts reading directly from Postgres (WHERE data_source='historical'
    # for historical figures; can union synthetic_daily to track the live stream, but
    # label the two sources separately in the chart).
    # TODO: scatter_geo via cities.latitude/longitude (joined through customers.city_id).
    st.info("TODO: monthly outstanding balance chart, default rate by grade/country, scatter_geo map.")
