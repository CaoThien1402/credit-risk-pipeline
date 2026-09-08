# CLAUDE.md

Credit risk & loan approval pipeline: Excel → PostgreSQL (4 normalized tables) → SQL feature view → XGBoost model → Streamlit app. Solo portfolio project, built session-by-session (see `docs/Credit_Risk_Pipeline_Plan_v2.md`).

Project memory (facts learned the hard way, dated): `docs/MEMORY.md`. Read it when picking up work on this repo.

**Stack**: Python 3.14 (via `uv`, `.venv/`), PostgreSQL 16 (Docker), SQLAlchemy 2.x, scikit-learn, XGBoost, SHAP, Streamlit, Plotly.

## Structure

```
sql/       schema.sql (DDL + rationale comments), feature_engineering.sql (CTE feature view)
etl/       historical_load.py, daily_ingest.py, config.py (DB connection),
           run_daily_ingest.ps1 (Windows Task Scheduler wrapper, see below)
data/      Credit_Risk_Dataset.xlsx — committed to the repo (not gitignored)
notebooks/ 01_eda.ipynb
models/    *.pkl joblib bundles — gitignored, not committed
app/       app.py (Streamlit, 2 tabs), utils.py
tests/     pytest, mirrors etl/ functions
```

## Commands

```bash
docker compose up -d
docker exec -i credit-db psql -U postgres -d credit_db -f sql/schema.sql   # or psql -h localhost if you have a client installed
python etl/historical_load.py    # run from repo root — SOURCE_FILE is cwd-relative
python etl/daily_ingest.py       # run from repo root
uv run pytest tests/             # pytest is a dev dependency, not in the main deps
streamlit run app/app.py
```

## Critical constraints

**Never add `loan_grade` or `loan_int_rate` to the at-application feature set** (ETL output used by the model, notebooks, `app.py`'s prediction tab). Verified on the real data: default rate runs ~10% at grade A to ~98% at grade G, and `loan_int_rate` is near-deterministic given grade. Both are outputs of an underwriting step that happens *after* the decision this model exists to make — including them is leakage that inflates AUC and produces a model nobody can actually run at application time. They're fine in `portfolio_risk_model.pkl`, a separate, explicitly-labeled model for analyzing the existing book.

**Never train on `data_source = 'synthetic_daily'` rows.** Those rows have `loan_status = NULL` (pending prediction) or a distribution-sampled label — not a real outcome either way. `sql/feature_engineering.sql` filters `WHERE data_source = 'historical'`; carry that filter into any new query against `loans`.

**`daily_ingest.py` UPSERTs on `application_ref`, not `loan_id`.** `loan_id` is a SERIAL Postgres assigns fresh on every insert, so `ON CONFLICT (loan_id)` never actually fires. `application_ref` is a hash of `client_id + loan_date`, computed in Python before insert — keep it that way if you touch the ingestion logic.

**Cap outliers before imputing, not after.** Real data has 5 rows with `person_age` 144/123 and 2 rows with `person_emp_length` 123 at age 21-22. `historical_load.py::clean_outliers` must run before `::impute_missing`, or the outliers pull the median before they're removed.

**Don't reintroduce `loan_to_income_ratio`.** Dropped in `drop_redundant_columns` — correlation with `loan_percent_income` is 0.9989 on the real data; keeping both just splits SHAP importance for one signal.

**`customers.age` is `NOT NULL`, but `clean_outliers` only caps it to `NaN` — it doesn't impute.** 5 real rows (age 144/123) hit this. `impute_missing()` median-imputes `person_age` too, alongside `emp_length`/`loan_int_rate`, as a stopgap so the historical load doesn't fail on the `NOT NULL` constraint. This was a judgment call (user-confirmed), not part of the original session 3-4 prompt — the real imputation strategy (median vs. KNN vs. group-median) is still open for session 8's EDA to revisit. Don't remove the `person_age` line from `impute_missing()` without replacing it with something else that keeps every row's age non-null before insert.

**The plan doc says "5 bảng" in a couple of places** (leftover from a v1 draft that had a separate `locations` table) **but the actual schema has 4**: `cities`, `customers`, `credit_bureau`, `loans`. Don't go looking for a 5th table — it doesn't exist and isn't needed.

## Conventions

- All Postgres access goes through `etl/config.py::get_engine()`, which reads `.env`. Never hardcode credentials. `config.py` calls `load_dotenv()` at import time — without it, every script raises `KeyError` on `os.environ['DB_USER']` etc. even though `.env` exists (this was actually missing and broke everything until fixed; don't remove it).
- Any script's `SOURCE_FILE`/relative data paths (e.g. `historical_load.py`'s `data/Credit_Risk_Dataset.xlsx`) resolve against **cwd**, not the script's own directory. Always run ETL scripts from the repo root.
- Windows' console defaults to cp1252. Any script whose `__main__` block prints Vietnamese text needs `sys.stdout.reconfigure(encoding="utf-8")`, or it crashes with `UnicodeEncodeError` on that print — after the actual DB work already succeeded. Don't mistake that crash for a data problem.
- `daily_ingest.py`'s `application_ref = hash(client_id + today)` means if the random sample happens to draw the same `client_id` twice in one day's batch, the second upsert silently overwrites the first — the day's row count can be a little less than `n` generated. Expected, not a bug.
- Model artifacts save as a bundle dict (`model`, `preprocessor`, `feature_names`, `trained_at`, `metrics`), never a bare model — `app.py` needs the preprocessor to transform new input consistently.
- Feature logic lives in `sql/feature_engineering.sql`, not duplicated in pandas. Read the file and run it via `pd.read_sql`; don't rewrite the joins in Python.

## Local machine state (not in git, won't exist after a fresh clone)

- Windows Task Scheduler job `CreditRiskPipeline_DailyIngest` — runs `etl/run_daily_ingest.ps1` daily at 20:00. Only fires while the user is logged into Windows; only succeeds if Docker Desktop is running at the time. Check/edit via `Get-ScheduledTask -TaskName CreditRiskPipeline_DailyIngest` or Task Scheduler GUI.
- GitHub remote: `https://github.com/CaoThien1402/credit-risk-pipeline` (public), default branch `main`.
- VSCode extension `ms-ossdata.vscode-pgsql` is the ad-hoc query tool in use here (not pgAdmin/DBeaver as the plan doc suggests) — already connected to `localhost:5432` / `credit_db`.

Full schema + rationale for every constraint: `sql/schema.sql`. Full session-by-session plan: `docs/Credit_Risk_Pipeline_Plan_v2.md`.
