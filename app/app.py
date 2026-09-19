"""Streamlit app with two tabs: new-application prediction and portfolio dashboard."""
# Two separate model bundles: at_application_model.pkl (Tab 1, no loan_grade/loan_int_rate)
# and portfolio_risk_model.pkl (Tab 2 only) — see the leakage note in sql/schema.sql.
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(REPO_ROOT))

from explanations import top_reasons
from utils import load_model_bundle
from model.features import ML_FEATURES_COLUMNS, get_feature_columns, split_numeric_categorical
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
    """Rebuild the SHAP explainer from the saved model at startup.

    Two options were available: (a) persist a fitted shap.TreeExplainer in the bundle
    next to model/preprocessor, or (b) reconstruct it here from bundle["model"].
    Chose (b). For a tree model TreeExplainer's construction is cheap (it reads the
    already-trained tree structure - there is no separate fit over data to redo), and
    st.cache_resource means it happens once per session, not once per prediction. (b)
    also leaves model/bundle.py's REQUIRED_KEYS contract untouched, so the existing
    bundles stay valid and test_bundle.py needs no new key; persisting a pickled
    explainer would instead add a key that every bundle must now carry, and pin a shap
    version into the artifact. Nothing about this app needs (a): the explainer is a
    pure function of the model, which is already in the bundle.

    Leading underscore tells st.cache_resource to skip hashing the model - it isn't
    trivially hashable, and this app only ever loads one at_application model per
    session, so there's no risk of silently reusing a stale explainer for a different
    model. Same TreeExplainer setup as notebooks/05_explainability_and_bundles.ipynb.
    """
    return shap.TreeExplainer(_model)


# --- Dashboard data (Tab 2) ------------------------------------------------------
# Queries live here as module constants so it's reviewable at a glance that the grade
# and country charts read dashboard_aggregates (never ml_features, never a Python
# re-aggregation), and that only the disbursement/geo charts touch raw tables, each
# for a reason stated in its own comment.

# default_rate_by_grade is already AVG(loan_status) OVER (PARTITION BY loan_grade) in
# sql/views.sql, so it's constant within a grade - MAX() just picks that constant out
# of the group. Deliberately not AVG(loan_status) here: recomputing in the app would
# duplicate feature logic that CLAUDE.md says lives in the view.
GRADE_QUERY = """
    SELECT loan_grade,
           MAX(default_rate_by_grade) AS default_rate,
           COUNT(*)                   AS n_loans,
           SUM(loan_amnt)             AS total_disbursed
    FROM dashboard_aggregates
    GROUP BY loan_grade
    ORDER BY loan_grade
"""

# No precomputed per-country window column exists in the view, so this aggregates in
# SQL - still not in pandas.
COUNTRY_QUERY = """
    SELECT country,
           AVG(loan_status::numeric) AS default_rate,
           COUNT(*)                  AS n_loans,
           SUM(loan_amnt)            AS total_disbursed
    FROM dashboard_aggregates
    GROUP BY country
    ORDER BY country
"""

# Raw `loans`, NOT dashboard_aggregates: that view inherits ml_features'
# WHERE data_source = 'historical', so it structurally cannot see the synthetic_daily
# stream this chart exists to track. data_source is selected so the two regimes can be
# drawn as separate series rather than silently summed into one trend line.
MONTHLY_QUERY = """
    SELECT date_trunc('month', loan_date)::date AS month,
           data_source,
           SUM(loan_amnt) AS total_disbursed,
           COUNT(*)       AS n_loans,
           MIN(loan_date) AS first_day,
           MAX(loan_date) AS last_day
    FROM loans
    GROUP BY 1, 2
    ORDER BY 1, 2
"""

# Raw tables again: latitude/longitude live on `cities` and are not carried into
# ml_features (they're not model features), so the map has to join for them.
GEO_QUERY = """
    SELECT ci.city,
           ci.country,
           ci.latitude::float8        AS latitude,
           ci.longitude::float8       AS longitude,
           COUNT(*)                   AS n_loans,
           AVG(l.loan_status::numeric) AS default_rate,
           SUM(l.loan_amnt)           AS total_disbursed
    FROM loans l
    JOIN customers c ON c.client_id = l.client_id
    JOIN cities ci   ON ci.city_id = c.city_id
    WHERE l.data_source = 'historical'
    GROUP BY ci.city, ci.country, ci.latitude, ci.longitude
    ORDER BY n_loans DESC
"""


@st.cache_resource
def get_db_engine():
    """Imported inside the function, not at module scope: etl.config calls load_dotenv()
    and reads DB credentials at import time, and Tab 1 must keep working with no .env
    and no database - scoring an application only needs the .pkl."""
    from etl.config import get_engine

    return get_engine()


@st.cache_data(ttl=300)
def run_dashboard_query(sql: str) -> pd.DataFrame:
    """Run one dashboard query, cached so switching tabs doesn't re-query.

    `sql` is a parameter rather than being read from a module constant inside the
    function body, so that it forms part of st.cache_data's key. With a no-argument
    loader, editing a query above leaves the function's own code unchanged, and
    Streamlit keeps serving the DataFrame from the previous query shape - which during
    development of this tab produced a KeyError on a column the new SQL does select.
    """
    with get_db_engine().connect() as conn:
        return pd.read_sql(sql, conn)


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

    # Source of truth for which fields the form shows is model/features.py, derived the
    # same way the training notebooks derive X's columns - not a hand-typed list, which
    # would silently go stale the moment features.py changed, and not bundle
    # ["feature_names"] alone, which only records what one saved artifact happens to
    # contain. Cross-checked against the bundle immediately below: the preprocessor was
    # fit on the bundle's columns in the bundle's order, so a disagreement means the
    # .pkl predates a features.py change and must be regenerated, not worked around.
    feature_names = get_feature_columns(list(ML_FEATURES_COLUMNS), "at_application")
    if feature_names != list(bundle["feature_names"]):
        st.error(
            "Feature mismatch between model/features.py and the saved model bundle.\n\n"
            f"- features.py expects: `{feature_names}`\n"
            f"- bundle was trained on: `{list(bundle['feature_names'])}`\n\n"
            "The bundle is stale. Re-run "
            "`notebooks/05_explainability_and_bundles.ipynb` to regenerate "
            "`models/at_application_model.pkl`."
        )
        st.stop()

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

            top_3 = pd.DataFrame(
                [
                    {
                        "Feature": name,
                        "Applicant's value": value,
                        "Effect": "increases risk" if contribution > 0 else "decreases risk",
                        "Contribution": round(contribution, 4),
                    }
                    for name, value, contribution in top_reasons(
                        row_shap_values, transformed_names, inputs, categorical_cols, n=3
                    )
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
    st.caption(
        "Describes the existing book, so unlike Tab 1 it does show `loan_grade` — "
        "grade is an underwriting output, which makes it leakage as a model input but "
        "legitimate as a reporting dimension."
    )

    try:
        grade_df = run_dashboard_query(GRADE_QUERY)
        country_df = run_dashboard_query(COUNTRY_QUERY)
        monthly_df = run_dashboard_query(MONTHLY_QUERY)
        geo_df = run_dashboard_query(GEO_QUERY)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        st.error(
            "Could not reach Postgres, so the dashboard has no data to show. "
            "Start it with `docker compose up -d` and reload.\n\n"
            f"```\n{type(exc).__name__}: {exc}\n```"
        )
        st.info("Tab 1 (New Application Prediction) does not need the database and still works.")
        st.stop()

    hist_loans = int(grade_df["n_loans"].sum())
    hist_disbursed = float(grade_df["total_disbursed"].sum())
    # Weighted by loans per grade - the mean of the 7 per-grade rates would weight
    # grade G's 64 loans the same as grade A's 10,777.
    overall_default_rate = float((grade_df["default_rate"] * grade_df["n_loans"]).sum() / hist_loans)
    synthetic_loans = int(monthly_df.loc[monthly_df["data_source"] == "synthetic_daily", "n_loans"].sum())

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Historical loans", f"{hist_loans:,}")
    kpi2.metric("Total disbursed", f"${hist_disbursed / 1e6:,.1f}M")
    kpi3.metric("Default rate", f"{overall_default_rate:.1%}")
    kpi4.metric("Daily-stream rows", f"{synthetic_loans:,}", help="data_source='synthetic_daily' — never used for training")

    st.divider()
    st.subheader("Monthly disbursement")

    monthly_df["month"] = pd.to_datetime(monthly_df["month"])
    hist_monthly = monthly_df[monthly_df["data_source"] == "historical"]
    synth_monthly = monthly_df[monthly_df["data_source"] == "synthetic_daily"]

    # Two explicit traces rather than one summed line: the synthetic stream is
    # generated data with a sampled/NULL label, so merging it into the historical
    # trend would present it as if it had the same statistical standing.
    fig_monthly = go.Figure()
    fig_monthly.add_bar(
        x=hist_monthly["month"], y=hist_monthly["total_disbursed"],
        name="historical", marker_color="#4C78A8",
    )
    fig_monthly.add_bar(
        x=synth_monthly["month"], y=synth_monthly["total_disbursed"],
        name="synthetic_daily", marker_color="#F58518",
    )

    # The series starts and ends mid-month, so the first and last bars are short for a
    # calendar reason, not a lending-volume reason. Annotating them is the difference
    # between a reader seeing a collapse and seeing a cut-off. Detected from the data
    # rather than hardcoded, so it stays correct as daily ingest extends the series.
    edges = {hist_monthly.index[0]: hist_monthly.iloc[0], hist_monthly.index[-1]: hist_monthly.iloc[-1]}
    for edge in edges.values():
        first_day, last_day = pd.to_datetime(edge["first_day"]), pd.to_datetime(edge["last_day"])
        if first_day.day == 1 and last_day == edge["month"] + pd.offsets.MonthEnd(0):
            continue  # a genuinely complete month; nothing to caveat
        # Day numbers built from .day rather than a strftime like %-d, which is
        # platform-specific and raises on Windows, where this app is developed.
        covered = f"{first_day:%b} {first_day.day}–{last_day.day}"
        fig_monthly.add_annotation(
            x=edge["month"], y=edge["total_disbursed"],
            text=f"partial month<br>({covered})",
            showarrow=True, arrowhead=2, ax=0, ay=-38, font=dict(size=11, color="#333"),
            bgcolor="rgba(255,255,255,0.92)", bordercolor="#999", borderwidth=1, borderpad=3,
        )
    fig_monthly.update_layout(
        barmode="group", height=420, xaxis_title=None,
        yaxis_title="Disbursed ($)", legend_title="data_source",
        margin=dict(t=30, b=10),
    )
    st.plotly_chart(fig_monthly, use_container_width=True)
    st.caption(
        f"Read from the raw `loans` table, not `dashboard_aggregates` — that view "
        f"inherits the `WHERE data_source = 'historical'` filter from `ml_features` "
        f"and so cannot see the {synthetic_loans} synthetic_daily rows at all. Those "
        "rows are a simulated live application stream (labels sampled or NULL); they "
        "are shown here to prove the ingest is running, and are never trained on."
    )

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Default rate by grade")
        fig_grade = px.bar(
            grade_df, x="loan_grade", y="default_rate",
            color="default_rate", color_continuous_scale="Reds",
            hover_data={"n_loans": ":,", "default_rate": ":.1%"},
        )
        fig_grade.update_layout(
            height=380, yaxis_tickformat=".0%", xaxis_title="loan_grade",
            yaxis_title="default rate", coloraxis_showscale=False, margin=dict(t=30, b=10),
        )
        st.plotly_chart(fig_grade, use_container_width=True)
        st.caption(
            "`default_rate_by_grade` comes precomputed from `dashboard_aggregates` "
            "(a window function in `sql/views.sql`), not re-derived here. The near-"
            "monotonic climb from A (10.0%) to G (98.4%) is exactly why grade is "
            "banned from the Tab 1 model: it encodes the underwriter's own risk "
            "verdict, which doesn't exist yet when a new application arrives."
        )

    with right:
        st.subheader("Default rate by country")
        # One flat colour on purpose: colouring by country would assign three loud,
        # different hues to three rates that are the same to within 0.13pp, which
        # reads as a difference the data doesn't contain.
        fig_country = px.bar(
            country_df, x="country", y="default_rate",
            hover_data={"n_loans": ":,", "default_rate": ":.2%"},
        )
        fig_country.update_traces(marker_color="#4C78A8")
        fig_country.update_layout(
            height=380, yaxis_tickformat=".0%", yaxis_range=[0, 0.3],
            xaxis_title=None, yaxis_title="default rate", showlegend=False,
            margin=dict(t=30, b=10),
        )
        st.plotly_chart(fig_country, use_container_width=True)
        spread_pp = (country_df["default_rate"].max() - country_df["default_rate"].min()) * 100
        st.caption(
            f"The three countries differ by {spread_pp:.2f} percentage points — a flat "
            "result, and reported as flat rather than rescaled to manufacture a visible "
            "gap. The y-axis is fixed to 0-30% for that reason. `country` is kept as a "
            "model feature for segmentation, not because it separates risk here."
        )

    st.divider()
    st.subheader("Portfolio by city")

    # range_color is fixed to 0-50% rather than left to autoscale. Autoscaling maps the
    # observed 20.5%-24.0% city spread across the full red-to-green ramp, so a 3.5pp
    # difference is drawn as if some cities were catastrophic and others pristine -
    # the same exaggeration the country chart's fixed y-axis avoids, and it would
    # contradict that chart sitting right above it. On a fixed scale the cities are
    # near-identical in colour, which is the actual finding; the map's real content is
    # bubble size, i.e. where the book is concentrated.
    fig_geo = px.scatter_geo(
        geo_df, lat="latitude", lon="longitude",
        size="n_loans", color="default_rate",
        color_continuous_scale="RdYlGn_r", range_color=[0.0, 0.5],
        hover_name="city",
        hover_data={"country": True, "n_loans": ":,", "default_rate": ":.1%",
                    "latitude": False, "longitude": False},
        size_max=28, projection="natural earth",
    )
    fig_geo.update_layout(height=520, margin=dict(t=10, b=10, l=0, r=0),
                          coloraxis_colorbar_title="default<br>rate",
                          coloraxis_colorbar_tickformat=".0%")
    fig_geo.update_geos(showcountries=True, countrycolor="#cccccc", showland=True, landcolor="#f5f5f5")
    st.plotly_chart(fig_geo, use_container_width=True)
    geo_lo, geo_hi = geo_df["default_rate"].min(), geo_df["default_rate"].max()
    st.caption(
        f"{len(geo_df)} cities, joined `loans` → `customers.city_id` → "
        "`cities.latitude/longitude`. Bubble size is loan count; colour is default "
        f"rate on a fixed 0–50% scale. Every city falls in a narrow "
        f"{geo_lo:.1%}–{geo_hi:.1%} band around the {overall_default_rate:.1%} book "
        "average, so the near-uniform colour is the result, not a rendering artefact. "
        "Historical rows only."
    )
