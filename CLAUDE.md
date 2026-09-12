# CLAUDE.md

Credit risk & loan approval pipeline: Excel → PostgreSQL (4 normalized tables) → SQL feature views → XGBoost model → Streamlit app. Solo portfolio project, built session-by-session (see `docs/Credit_Risk_Pipeline_Plan_v3.md` — v3 supersedes v2, which has been deleted; don't recreate it).

Project memory (facts learned the hard way, dated): `docs/MEMORY.md`. Read it when picking up work on this repo.

**Stack**: Python 3.14 (via `uv`, `.venv/`), PostgreSQL 16 (Docker), SQLAlchemy 2.x, scikit-learn, XGBoost, SHAP, Streamlit, Plotly. Dev-only deps (`uv add --dev`, not in main deps): `pytest`, `jupyter`, `kaleido` (static PNG export from Plotly figures).

## Structure

```
sql/       schema.sql (DDL + rationale comments), views.sql (ml_features — base columns
           for training; dashboard_aggregates — ml_features + window-function columns
           for Streamlit only. Two separate views, not one query — see Critical constraints.)
etl/       historical_load.py, daily_ingest.py, config.py (DB connection),
           run_daily_ingest.ps1 (Windows Task Scheduler wrapper, see below)
data/      Credit_Risk_Dataset.xlsx — committed to the repo (not gitignored)
notebooks/ 01_eda.ipynb, 02_baseline_model.ipynb (session 9+), figures/ (PNGs exported
           for the README, via kaleido)
model/     features.py (column bookkeeping: ID/target/leakage cols, categorical list),
           preprocessing.py (ColumnTransformer + Pipeline — written by hand in session 9,
           not AI-generated; see docs/Credit_Risk_Pipeline_Plan_v3.md buổi 9). Reused
           as-is by sessions 10-12, saved as `preprocessor` in the session 12 bundle.
models/    *.pkl joblib bundles — gitignored, not committed
app/       app.py (Streamlit, 2 tabs), utils.py
tests/     pytest — etl/ function tests, model/ column-split + preprocessing-discipline
           tests, bundle-contract tests, and SQL structural tests (test_views.py,
           needs a reachable Postgres — skips otherwise; CI's postgres service and
           schema.sql/views.sql apply step make it always run there)
scripts/   dump_db.sh — snapshots the running DB into db-seed/
db-seed/   01_seed.sql — mounted at /docker-entrypoint-initdb.d, Postgres auto-loads it
           on an empty volume; refresh via scripts/dump_db.sh after changing the data
```

## Commands

```bash
docker compose up -d             # db-seed/01_seed.sql auto-loads on an empty volume —
                                  # no need to re-run the ETL just to get data back
docker exec -i credit-db psql -U postgres -d credit_db -f - < sql/schema.sql   # only needed if rebuilding from scratch (empty db-seed/)
docker exec -i credit-db psql -U postgres -d credit_db -f - < sql/views.sql    # ditto — views, re-run only if their definitions change
# Note the `-f - < file` form: `-f sql/schema.sql` alone looks for that path INSIDE the
# container via `docker exec`, which only has ./db-seed mounted (see docker-compose.yml)
# — it 404s. `-f -` reads from stdin, which the host-side `< file` redirect supplies.
# Verified 2026-09-12: `docker exec -i credit-db psql ... -f sql/schema.sql` fails with
# "No such file or directory". (CI runs psql directly on the runner, not via docker exec,
# so sql/schema.sql is a real path there and `-f sql/schema.sql` works as-is — the -f -
# form is specifically for the docker exec case.)
python etl/historical_load.py    # run from repo root — SOURCE_FILE is cwd-relative
python etl/daily_ingest.py       # run from repo root
bash scripts/dump_db.sh          # refresh db-seed/01_seed.sql after changing the data
uv run pytest tests/             # pytest is a dev dependency, not in the main deps
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_baseline_model.ipynb
streamlit run app/app.py
```

## Critical constraints

**Never add `loan_grade` or `loan_int_rate` to the at-application feature set** (ETL output used by the model, notebooks, `app.py`'s prediction tab). Verified on the real data: grade is a near-total predictor of default, and `loan_int_rate` is near-deterministic given grade (numbers in `docs/MEMORY.md`). Both are outputs of an underwriting step that happens *after* the decision this model exists to make — including them is leakage that inflates AUC and produces a model nobody can actually run at application time. They're fine in `portfolio_risk_model.pkl`, a separate, explicitly-labeled model for analyzing the existing book.

**Training code queries `ml_features`, never `dashboard_aggregates`.** `sql/views.sql` splits these into two separate Postgres views specifically so this isn't just a comment to remember — `dashboard_aggregates` (session 15-16 Streamlit only) adds three window-function columns (`avg_loan_amnt_by_age_bucket`, `default_rate_by_grade`, `util_rank_in_country`) on top of `ml_features`. `default_rate_by_grade` is literally `AVG(loan_status)` — the target itself — computed over the whole historical table before any train/test split; using it as a feature is a worse leak than `loan_grade` alone. The other two are computed over the full population too (not per-fold), so they leak test-set distribution into training. The view split is the primary safeguard; this line is a reminder, not the only one.

**Never train on `data_source = 'synthetic_daily'` rows.** Those rows have `loan_status = NULL` (pending prediction) or a distribution-sampled label — not a real outcome either way. `ml_features` (in `sql/views.sql`) filters `WHERE data_source = 'historical'`; carry that filter into any new query against `loans`.

**`daily_ingest.py` UPSERTs on `application_ref`, not `loan_id`.** `loan_id` is a SERIAL Postgres assigns fresh on every insert, so `ON CONFLICT (loan_id)` never actually fires. `application_ref` is a hash of `client_id + loan_date`, computed in Python before insert — keep it that way if you touch the ingestion logic.

**Cap outliers before imputing, not after.** Real data has rows with impossible `person_age` values and `person_emp_length` values inconsistent with age (exact rows/values in `docs/MEMORY.md`). `historical_load.py::clean_outliers` must run before `::impute_missing`, or the outliers pull the median before they're removed.

**Don't reintroduce `loan_to_income_ratio`.** Dropped in `drop_redundant_columns` — near-perfectly correlated with `loan_percent_income` on the real data (exact correlation in `docs/MEMORY.md`); keeping both just splits SHAP importance for one signal.

**`customers.age` is `NOT NULL`, but `clean_outliers` only caps it to `NaN` — it doesn't impute.** A handful of real rows hit this (see `docs/MEMORY.md` for the exact rows/values). `impute_missing()` median-imputes `person_age` too, alongside `emp_length`/`loan_int_rate`, as a stopgap so the historical load doesn't fail on the `NOT NULL` constraint. This was a judgment call (user-confirmed), not part of the original session 3-4 prompt. Session 8's EDA confirmed median imputation is reasonable: both missingness patterns look MCAR, not MAR — no grade/group signal a smarter imputer would exploit (numbers in `docs/MEMORY.md`). Don't remove the `person_age` line from `impute_missing()` without replacing it with something else that keeps every row's age non-null before insert.

**The schema has 4 tables**: `cities`, `customers`, `credit_bureau`, `loans`. This has been mis-stated as "5 bảng" twice now — once in the original v1 draft (had a separate `locations` table), and again when v3 of the plan was drafted from scratch in a different chat that didn't know about the v2 fix. If the plan doc gets regenerated or edited externally again, re-check this number before trusting it.

**Session 9+ preprocessing must use `sklearn.pipeline.Pipeline` + `ColumnTransformer`, fit only on the train split — this is a requirement to implement, not a description of finished code (see the stub note at the end of this bullet).** `OneHotEncoder` for nominal categoricals (`model/features.py::NOMINAL_CATEGORICAL_COLS`) — not `LabelEncoder`, which imposes a false ordinal relationship a linear model can misread. `StandardScaler` for numeric columns, inside the same `ColumnTransformer`, fit via `Pipeline.fit(X_train, ...)` *after* `train_test_split` — fitting on the full dataset first leaks test-set statistics into training. `model/preprocessing.py::build_preprocessor`/`build_pipeline` are left as `NotImplementedError` stubs on purpose — write them by hand (see that file's docstring), don't generate them wholesale; it's reused as-is for session 10-11 (XGBoost) and saved as `preprocessor` in the session 12 model bundle.

## Conventions

- All Postgres access goes through `etl/config.py::get_engine()`, which reads `.env`. Never hardcode credentials. `config.py` calls `load_dotenv()` at import time — without it, every script raises `KeyError` on `os.environ['DB_USER']` etc. even though `.env` exists (this was actually missing and broke everything until fixed; don't remove it).
- Any script's `SOURCE_FILE`/relative data paths (e.g. `historical_load.py`'s `data/Credit_Risk_Dataset.xlsx`) resolve against **cwd**, not the script's own directory. Always run ETL scripts from the repo root.
- Windows' console defaults to cp1252. Any script whose `__main__` block prints Vietnamese text needs `sys.stdout.reconfigure(encoding="utf-8")`, or it crashes with `UnicodeEncodeError` on that print — after the actual DB work already succeeded. Don't mistake that crash for a data problem.
- `daily_ingest.py`'s `application_ref = hash(client_id + today)` means if the random sample happens to draw the same `client_id` twice in one day's batch, the second upsert silently overwrites the first — the day's row count can be a little less than `n` generated. Expected, not a bug.
- Model artifacts save as a bundle dict (`model`, `preprocessor`, `feature_names`, `trained_at`, `metrics`), never a bare model — `app.py` needs the preprocessor to transform new input consistently.
- Feature logic lives in `sql/views.sql`, not duplicated in pandas. Query `ml_features`/`dashboard_aggregates` via `pd.read_sql("SELECT * FROM ml_features", engine)`; don't rewrite the joins in Python.

## Git workflow (adopted 2026-09-09)

Every change goes through a feature branch → PR → self-review → merge, even solo —
this is a deliberate habit-building choice (common in real DE/DS job descriptions),
not overhead to skip. Never commit straight to `main`. Concretely:

1. `git checkout -b <type>/<short-description>` (e.g. `feat/session-9-baseline`)
2. Commit, push the branch, open a PR with `gh pr create` and a clear description
   (what changed, why, how it was verified)
3. Self-review the diff (`gh pr diff`) — GitHub won't let the author formally
   "Approve" their own PR, so this is a read-through, not a clicked approval
4. Merge (`gh pr merge --squash --delete-branch`) once CI is green

CI (`.github/workflows/ci.yml`) runs `pytest` via GitHub Actions on every push and
on PRs to `main`. A red check is a signal to fix before merging, not to ignore.

**PR hygiene** (adopted 2026-09-12, generalized from an unrelated project's contributor
guide found in this repo's folder — most of that guide didn't apply here, but these
three habits do):

- **Disclose AI involvement.** Every commit ends with
  `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`; every PR description ends
  with the Claude Code attribution line. Never omit these when Claude Code authored or
  co-authored the change.
- **One change per PR.** Don't bundle an unrelated fix into a feature branch just
  because you noticed it along the way — split it into its own branch/PR (e.g. PR #6
  split a cosmetic notebook re-save and an unrelated editor-config fix out of PR #5).
- **Verify before claiming done.** A fix isn't finished when the code is written — run
  it: a query against the live DB, `pytest`, a notebook re-execution via `nbconvert`.
  Don't open a PR whose "Test plan" section describes checks that were never actually
  run.

## Local machine state (not in git, won't exist after a fresh clone)

- Windows Task Scheduler job `CreditRiskPipeline_DailyIngest` — runs `etl/run_daily_ingest.ps1` daily at 20:00. Only fires while the user is logged into Windows; only succeeds if Docker Desktop is running at the time. Check/edit via `Get-ScheduledTask -TaskName CreditRiskPipeline_DailyIngest` or Task Scheduler GUI.
- GitHub remote: `https://github.com/CaoThien1402/credit-risk-pipeline` (public), default branch `main`.
- VSCode extension `ms-ossdata.vscode-pgsql` is the ad-hoc query tool in use here (not pgAdmin/DBeaver as the plan doc suggests) — already connected to `localhost:5432` / `credit_db`.

Full schema + rationale for every constraint: `sql/schema.sql`. Full session-by-session plan: `docs/Credit_Risk_Pipeline_Plan_v3.md`.
