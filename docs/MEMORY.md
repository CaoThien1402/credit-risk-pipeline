# MEMORY.md

Facts and patterns discovered while working on this repo. Append here when something is learned the hard way — a wrong assumption caught, a number worth not re-deriving. Keep entries short and dated. This file is read in full each session (first 200 lines); prune anything superseded rather than letting it accumulate.

## 2026-09-08 — initial data audit (direct read of Credit_Risk_Dataset.xlsx)

- Shape confirmed: 32,581 rows × 29 columns, 3 countries, default rate 21.82%. Matches the original plan's claims.
- Missing values only in `person_emp_length` (895 rows) and `loan_int_rate` (3,116 rows) — no other column has nulls.
- `person_age`: 5 rows at 144 or 123 (impossible). `person_emp_length`: 2 rows at 123 years with age 21-22 in the same row. Both are the well-known error pattern in this dataset lineage, not a hypothesis — cap, don't just flag.
- `loan_grade` → default rate: A=9.96%, B=16.3%, C=20.7%, D=59%, E=64%, F=70.5%, G=98.4%. Near-monotonic and near-total separation — this is why it's excluded from the at-application feature set (see CLAUDE.md).
- `loan_int_rate` within each grade has std ≈1%, means climb roughly 7.3→20.25 from A→G. Grade essentially determines rate; don't use both as independent at-application features even if grade were allowed.
- `loan_percent_income` vs `loan_to_income_ratio`: corr = 0.9989, mean absolute difference 0.0028. Same signal computed twice with different rounding — keep only `loan_percent_income`.
- Exactly 18 distinct `(country, state, city)` combinations in the whole dataset, and lat/long is 1:1 with city (deterministic). That's why `cities` is a separate dimension table instead of repeating coordinates per customer row.
- No date or timestamp column anywhere in the 29 source columns. `loan_date` in Postgres is synthetically backfilled over a 24-month window at load time — it is not a real origination date. Don't present it to a user as historical fact without that caveat; the dashboard prompt in the plan already carries this note, keep it if the prompt is edited.
- `client_id` is unique per row (32,581 unique values for 32,581 rows) — the source data is 1:1 customer:loan even though the schema supports 1:many going forward.

## 2026-09-08 — sessions 1-7: environment setup, ETL (buổi 1-5), CLAUDE.md audit

- Local dev environment actually runs **Python 3.14.3 via `uv`** (`.venv/`), not 3.11 — `CLAUDE.md` said 3.11 until this session corrected it. Docker Desktop + `credit-db` (Postgres 16) container. VSCode extension `ms-ossdata.vscode-pgsql` is the query tool in use, not pgAdmin/DBeaver.
- `etl/config.py` didn't call `load_dotenv()` when the scaffold was first inspected — every script raised `KeyError` on `os.environ['DB_USER']` despite `.env` existing and `python-dotenv` being a listed dependency. Fixed by adding `load_dotenv()` at module import time.
- `customers.age` is `NOT NULL`, but `historical_load.py::clean_outliers` only caps `person_age > 100` to `NaN` — it never imputes. 5 real rows hit this and would fail the `NOT NULL` insert. Decided (user-confirmed) to median-impute `person_age` in `impute_missing()` too, as a stopgap alongside `emp_length`/`loan_int_rate`; buổi 8 EDA later confirmed this was a reasonable choice (see below).
- `historical_load.py`'s `SOURCE_FILE` is `data/Credit_Risk_Dataset.xlsx`, resolved against **cwd** — scripts must run from the repo root, not from inside `etl/`.
- Windows console is cp1252 by default; scripts that `print()` Vietnamese text need `sys.stdout.reconfigure(encoding="utf-8")` in `__main__`, or they crash with `UnicodeEncodeError` on the last line — after the DB work already succeeded. Fixed in both `historical_load.py` and `daily_ingest.py`.
- `daily_ingest.py`'s `application_ref = hash(client_id + today)`: if a day's random sample draws the same `client_id` twice, the second upsert overwrites the first — that day's stored row count can be a little less than `n` generated. Expected, verified via fixed-seed re-run test (row count stayed flat across two identical-seed runs).
- The plan doc says "5 bảng" in a couple of places (leftover from a v1 draft with a separate `locations` table) but only ever needs 4: `cities`, `customers`, `credit_bureau`, `loans`.
- `sql/feature_engineering.sql` was already fully written in the scaffold (buổi 6 CTE + buổi 7 window functions both present, unlike the `etl/*.py` files which had real `NotImplementedError` TODOs) — verified rather than authored. Buổi 6 check passed: CTE returns exactly 32,581 rows (== `historical` loan count), 0 with `loan_status IS NULL` (confirms no `synthetic_daily` leakage).
- `pytest` was referenced in the plan/README's run commands but wasn't an actual dependency anywhere, and `tests/test_placeholder.py` had `sys.path.insert(0, "../etl")` — resolves outside the repo when run from root. Added `pytest` as a `uv` dev dependency and fixed the path to be relative to `__file__`; both tests pass now.
- Root `.gitignore` said `venv/` but the real folder is `.venv/` — harmless in practice only because `uv` writes its own `.venv/.gitignore` (`*`) inside the venv itself. Fixed the root pattern anyway; don't rely on the inner file.
- Windows Task Scheduler job `CreditRiskPipeline_DailyIngest` (daily 20:00, via `etl/run_daily_ingest.ps1`) is registered on this machine only — it is not tracked in git and won't exist after a fresh clone or on another machine. Only fires while logged into Windows, only succeeds if Docker is running.
- Repo pushed to `https://github.com/CaoThien1402/credit-risk-pipeline` (public), default branch renamed `master` → `main`. `data/Credit_Risk_Dataset.xlsx` is committed to the repo — an explicit user choice, not the general best practice default.

## 2026-09-08 — buổi 8: EDA (notebooks/01_eda.ipynb)

- By the time data reaches Postgres, `loan_int_rate`/`emp_length`/`age` already have **zero
  nulls** — `historical_load.py::impute_missing` ran before insert. To check the *original*
  missingness pattern, the notebook loads `data/Credit_Risk_Dataset.xlsx` a second time,
  separately from the `feature_engineering.sql` read.
- `loan_int_rate` missing rate by `loan_grade`: A=9.3%, B=10.1%, C=9.8%, D=8.6%, E=8.6%,
  F=11.2%, G=7.8% — flat, no trend.
- `person_emp_length` missing rate by `employment_type`: Full-time=2.8%, Part-time=2.8%,
  Self-employed=2.8%, **Unemployed=2.2%** (lowest, not highest — the intuitive guess that
  unemployed applicants would be missing emp_length more often is wrong). Both patterns
  read as **MCAR**, not MAR — median imputation (already applied) is a reasonable choice,
  nothing more sophisticated is obviously justified by the missingness mechanism.
- Cross-checked post-ETL: 0 rows with `age > 100` in Postgres, 0 rows violating
  `emp_length <= age - 14`, age range 20-94. Confirms `clean_outliers` + `impute_missing`
  did their job on the full 32,581-row load, not just the sample checked back in session 4.
- `jupyter` and `kaleido` added as `uv` dev dependencies — neither was installed despite the
  README instructing `jupyter notebook notebooks/01_eda.ipynb`. `kaleido` is what makes
  `fig.write_image(...)` work for exporting static PNGs (used to save 3 figures into
  `notebooks/figures/` for later README embedding in buổi 17).
- Notebook executed end-to-end via `jupyter nbconvert --to notebook --execute --inplace`
  (12 code cells, 0 errors) rather than just eyeballed — the committed `.ipynb` has real
  output, not just source.

## Open questions — not yet resolved

- `income` (max ~6,000,000) and `other_debt` (max ~1,190,000) have heavy right tails. Not yet determined whether these are genuine high earners or data errors — currently left uncapped. If model calibration looks off in the tails during buổi 9-11, revisit this before assuming the model is at fault.
