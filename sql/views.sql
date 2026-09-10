-- ml_features: base columns only, for training/notebooks. WHERE data_source = 'historical'
-- keeps only rows with a real outcome, excluding synthetic_daily applications
-- (loan_status is NULL/sampled for those).
-- Includes every fully-populated customers/credit_bureau/loans column that isn't an
-- identifier, a leakage column, or a dashboard-only aggregate (see CLAUDE.md) — gender,
-- marital_status, education_level, employment_type, open_accounts, other_debt, and
-- loan_term_months were missing from the original buổi 6-7 CTE despite being 100%
-- populated in the real data; added here so session 9's feature set isn't short 6 columns
-- the plan explicitly calls for.
CREATE OR REPLACE VIEW ml_features AS
SELECT
    l.loan_id,
    l.client_id,
    l.loan_intent,
    l.loan_grade,
    l.loan_amnt,
    l.loan_int_rate,
    l.loan_percent_income,
    l.debt_to_income_ratio,
    l.loan_status,
    c.age,
    c.income,
    c.home_ownership,
    c.emp_length,
    cb.default_on_file,
    cb.cred_hist_length,
    cb.credit_utilization_ratio,
    cb.past_delinquencies,
    ci.country,
    -- appended after the original buổi 6-7 columns above (Postgres CREATE OR REPLACE VIEW
    -- can only add columns at the end, not reorder them) — see comment above.
    l.loan_term_months,
    l.other_debt,
    c.gender,
    c.marital_status,
    c.education_level,
    c.employment_type,
    cb.open_accounts
FROM loans l
JOIN customers c      ON c.client_id = l.client_id
JOIN credit_bureau cb ON cb.client_id = l.client_id
LEFT JOIN cities ci   ON ci.city_id = c.city_id
WHERE l.data_source = 'historical';

-- dashboard_aggregates: ml_features plus the 3 window-function aggregates used only by
-- the session 15-16 Streamlit dashboard. Structurally separate from ml_features so
-- training code has no path to select these columns by accident — see CLAUDE.md.
-- Uses SELECT * from ml_features, so adding a column to ml_features shifts this view's
-- column positions. CREATE OR REPLACE VIEW cannot handle that (Postgres only allows
-- appending columns at the end, not reordering) — if it errors with "cannot change name
-- of view column", run DROP VIEW dashboard_aggregates first, then re-apply this file.
CREATE OR REPLACE VIEW dashboard_aggregates AS
SELECT
    *,
    AVG(loan_amnt) OVER (PARTITION BY (age / 10) * 10) AS avg_loan_amnt_by_age_bucket,
    AVG(loan_status::numeric) OVER (PARTITION BY loan_grade) AS default_rate_by_grade,
    RANK() OVER (PARTITION BY country ORDER BY credit_utilization_ratio DESC) AS util_rank_in_country
FROM ml_features;
