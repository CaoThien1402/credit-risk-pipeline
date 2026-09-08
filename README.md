# Credit Risk & Loan Approval Pipeline

Scaffold for the 17-session plan in `docs/Credit_Risk_Pipeline_Plan_v2.md`, with fixes already
integrated from a direct audit of `Credit_Risk_Dataset.xlsx` (32,581 rows, 29 columns).

## Differences from the original plan

| # | Issue | Fixed in |
|---|---|---|
| 1 | `loan_grade`/`loan_int_rate` are near-deterministic with the target → leakage if used as input for the new-application form | `app/app.py`, split into 2 model bundles |
| 2 | 5 rows with `person_age`=144/123, 2 rows with `person_emp_length`=123 at age 21-22 | `etl/historical_load.py::clean_outliers`, `sql/schema.sql` CHECK constraints |
| 3 | `loan_percent_income` and `loan_to_income_ratio` are duplicates (corr=0.9989) | `etl/historical_load.py::drop_redundant_columns` |
| 4 | Source data has no date column → monthly dashboard has no history | `etl/historical_load.py::backfill_loan_dates` |
| 5 | `ON CONFLICT (loan_id)` doesn't work because `loan_id` is SERIAL | `sql/schema.sql` (`application_ref`), `etl/daily_ingest.py` |
| 6 | Daily synthetic applications leak into the training set if not filtered | `sql/schema.sql` (`data_source`), `sql/feature_engineering.sql` |
| 7 | A `locations` table would repeat coordinates 32,581 times for only 18 real city combos | `sql/schema.sql` (separate `cities` table) |

## Structure

```
credit-risk-pipeline/
├── sql/
│   ├── schema.sql              # DDL — see the comments in the file for rationale
│   └── feature_engineering.sql # CTE + window functions
├── etl/
│   ├── config.py                # Postgres connection from .env
│   ├── historical_load.py       # bulk-loads the historical dataset
│   └── daily_ingest.py          # simulates new daily applications
├── notebooks/
│   └── 01_eda.ipynb             # EDA, with a checklist
├── models/                      # model bundles (.pkl) saved here
├── app/
│   ├── app.py                   # Streamlit, 2 tabs
│   └── utils.py
├── tests/
│   └── test_placeholder.py      # tests for the key ETL functions
├── docker-compose.yml           # Postgres
├── requirements.txt
└── .env.example
```

## Running from scratch

```bash
cp .env.example .env               # set a real password
docker compose up -d
psql -h localhost -U postgres -d credit_db -f sql/schema.sql
python etl/historical_load.py
python etl/daily_ingest.py
jupyter notebook notebooks/01_eda.ipynb
streamlit run app/app.py
```
