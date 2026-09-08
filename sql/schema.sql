-- Credit Risk Pipeline schema.
-- Design notes for each table are commented directly below it.

CREATE TABLE cities (
    city_id     SERIAL PRIMARY KEY,
    country     VARCHAR(50) NOT NULL,
    state       VARCHAR(50) NOT NULL,
    city        VARCHAR(50) NOT NULL,
    latitude    NUMERIC(9,6) NOT NULL,
    longitude   NUMERIC(9,6) NOT NULL,
    UNIQUE (country, state, city)
);
-- Only 18 distinct (country, state, city) combos exist in the data, each with a fixed
-- lat/long — a separate dimension table avoids repeating coordinates per customer
-- (32,581 rows collapse to 18) and keeps the schema in 3NF.

CREATE TABLE customers (
    client_id       VARCHAR(20) PRIMARY KEY,
    age             SMALLINT NOT NULL CHECK (age BETWEEN 18 AND 100),
    income          NUMERIC(14,2) NOT NULL CHECK (income >= 0),
    home_ownership  VARCHAR(20) NOT NULL CHECK (home_ownership IN ('RENT','OWN','MORTGAGE','OTHER')),
    emp_length      NUMERIC(5,2) CHECK (emp_length IS NULL OR (emp_length >= 0 AND emp_length <= age - 14)),
    gender          VARCHAR(10),
    marital_status  VARCHAR(20),
    education_level VARCHAR(20),
    employment_type VARCHAR(20),
    city_id         INT REFERENCES cities(city_id),
    created_at      TIMESTAMP NOT NULL DEFAULT now()
);
-- age BETWEEN 18 AND 100: real data has 5 rows with age 144/123, a known data-entry error.
-- emp_length <= age - 14: real data has 2 rows with emp_length 123 at age 21-22 (impossible).
-- city_id is a plain FK, not a join table — customer-to-city is 1:1 at any point in time.

CREATE TABLE credit_bureau (
    client_id                 VARCHAR(20) PRIMARY KEY REFERENCES customers(client_id),
    default_on_file           CHAR(1) NOT NULL CHECK (default_on_file IN ('Y','N')),
    cred_hist_length          SMALLINT NOT NULL CHECK (cred_hist_length >= 0),
    open_accounts             SMALLINT CHECK (open_accounts >= 0),
    credit_utilization_ratio  NUMERIC(5,4) CHECK (credit_utilization_ratio BETWEEN 0 AND 1),
    past_delinquencies        SMALLINT CHECK (past_delinquencies >= 0)
);

CREATE TABLE loans (
    loan_id             SERIAL PRIMARY KEY,
    application_ref     VARCHAR(64) NOT NULL UNIQUE,
    client_id           VARCHAR(20) NOT NULL REFERENCES customers(client_id),
    loan_intent         VARCHAR(30),
    loan_grade          CHAR(1) CHECK (loan_grade IN ('A','B','C','D','E','F','G')),
    loan_amnt           NUMERIC(14,2) NOT NULL CHECK (loan_amnt > 0),
    loan_int_rate       NUMERIC(5,2),
    loan_term_months    SMALLINT CHECK (loan_term_months IN (12,24,36,60)),
    loan_status         SMALLINT CHECK (loan_status IN (0,1)),  -- NULL = synthetic application awaiting a model prediction
    loan_percent_income NUMERIC(6,4),
    other_debt          NUMERIC(14,2),
    debt_to_income_ratio NUMERIC(6,4),
    loan_date           DATE NOT NULL,
    data_source         VARCHAR(16) NOT NULL DEFAULT 'historical'
                         CHECK (data_source IN ('historical','synthetic_daily')),
    created_at          TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX idx_loans_client ON loans(client_id);
CREATE INDEX idx_loans_date   ON loans(loan_date);
CREATE INDEX idx_loans_source ON loans(data_source);

-- Key design decisions:
-- 1) application_ref (UNIQUE, generated in ETL) is the UPSERT key, not loan_id (SERIAL) —
--    a SERIAL always gets a fresh value, so ON CONFLICT on it would never fire.
-- 2) loan_to_income_ratio is dropped (corr = 0.9989 with loan_percent_income) — the same
--    signal computed twice just inflates multicollinearity and splits SHAP importance.
-- 3) loan_date + data_source are synthetic additions — the source data has no date column
--    at all; loan_date is backfilled so monthly trend charts have history to show.
-- 4) data_source lets queries filter to WHERE data_source = 'historical' when training,
--    excluding synthetic rows whose status is sampled, not a real outcome.
-- 5) loan_status allows NULL — a freshly submitted application has no outcome yet;
--    predicting it is the model's job.

-- Feature leakage note (applies to model training, not this DDL):
-- loan_grade and loan_int_rate are near-deterministic with respect to the target. The
-- at-application model must not use them as input; only the portfolio-risk model may,
-- and that distinction must be documented in the README.
