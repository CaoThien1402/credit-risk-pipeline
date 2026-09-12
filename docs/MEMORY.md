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

## 2026-09-09 — plan correction: window-function columns are not model features

- User asked why buổi 9's prompt talks about "raw" customer/credit_bureau info when
  Postgres already holds cleaned data — root cause was an undocumented gap, not a
  misunderstanding: `sql/feature_engineering.sql`'s 3 window columns
  (`avg_loan_amnt_by_age_bucket`, `default_rate_by_grade`, `util_rank_in_country`) were
  built for the buổi 15-16 dashboard, but the plan never said "don't feed these into a
  model." `default_rate_by_grade` is `AVG(loan_status)` over the whole historical table
  — a direct target leak, worse than `loan_grade` alone, since it's computed before any
  train/test split exists.
- Fixed in `docs/Credit_Risk_Pipeline_Plan_v2.md`: added a warning after buổi 6-7's
  result line, and prepended an explicit "select base columns only" instruction to buổi
  9's prompt. Also added as a `CLAUDE.md` critical constraint.
- Not a code bug — no session 9-11 code exists yet, so nothing needed fixing in `etl/`
  or notebooks. Purely a plan/doc gap caught before it could produce a real leak.

## 2026-09-09 — plan v3 adopted, db-seed infra, generalized leakage scan

- User got a v3 rewrite of the plan from a separate Claude chat
  (`docs/Credit_Risk_Pipeline_Plan_v3.md`) and asked to reconcile it. v3's real
  content deltas: (a) a portable DB-seed workflow via
  `docker-entrypoint-initdb.d`, (b) buổi 8 rewritten to teach a general leakage-
  detection *method* instead of stating the dataset's specific numbers upfront
  (to avoid `CLAUDE.md`/`MEMORY.md` "spoiling" the discovery for someone starting
  fresh), (c) a rule to delay adding `CLAUDE.md`/`MEMORY.md` to a fresh repo until
  buổi 10, for the same reason.
- **v3 reintroduced the "5 bảng" bug** that was fixed in v2 — it was drafted from
  scratch in a different chat with no access to that earlier fix. Re-applied the
  same 7-spot correction (4 tables: `cities`, `customers`, `credit_bureau`,
  `loans`). If the plan doc gets externally regenerated again, check this first.
- **`docs/Credit_Risk_Pipeline_Plan_v2.md` was deleted** (by the user, outside
  the chat) and **v3 is now the canonical plan** — updated `README.md` and
  `CLAUDE.md` to point at v3. Old entries in this file that still say "v2" are
  historical and were true when written; they're not being rewritten.
- **Did not apply** v3's "delay `CLAUDE.md`/`MEMORY.md` to buổi 10" rule
  retroactively (user's explicit decision) — buổi 8 in this repo was already
  computed for real (the MCAR finding even contradicted the initial guess about
  `Unemployed`/`emp_length`), so there was nothing to "spoil." That rule is for
  someone starting the plan fresh, not applicable mid-project.
- **Implemented the db-seed workflow**: `docker-compose.yml` now mounts
  `./db-seed:/docker-entrypoint-initdb.d`; `scripts/dump_db.sh` runs
  `pg_dump --clean --if-exists` into `db-seed/01_seed.sql`. Verified end-to-end:
  `docker compose down -v && docker compose up -d` restores all 4 tables and the
  full 32,581-row historical dataset with no ETL run. `db-seed/01_seed.sql` is
  ~9MB, committed (not gitignored) — same call as committing the raw dataset.
- **Extended `notebooks/01_eda.ipynb`** with v3's generalized leakage-scan
  method: default-rate spread across *every* categorical column
  (`loan_grade`=88.5 points, next-highest `home_ownership`=24.1, `country`≈0) and
  within-group variance ratio for every numeric-categorical pair (`loan_grade`→
  `loan_int_rate`=0.39, next-lowest pair=0.62). `loan_grade`/`loan_int_rate` top
  both scans by a wide margin — the general method reproduces the session 8
  conclusion without hardcoding which columns to check. Re-executed, 14 code
  cells, 0 errors.
- Added a `CLAUDE.md` constraint for session 9+: preprocessing must use
  `sklearn.pipeline.Pipeline` + `ColumnTransformer` (`OneHotEncoder` for nominal
  categoricals, `StandardScaler` for numerics), fit only on the train split —
  this came from a separate curriculum note the user is following (interview-
  prep: explain *why* scaling happens after the split and why not
  `LabelEncoder`), not from the plan doc itself.

## 2026-09-10 — sql/feature_engineering.sql split into two views

- Replaced the single ad-hoc CTE query (`sql/feature_engineering.sql`, read from
  disk by `pd.read_sql`) with `sql/views.sql`, two real Postgres views:
  `ml_features` (the old "base" CTE, 18 columns, `WHERE data_source='historical'`)
  and `dashboard_aggregates` (`ml_features` + the 3 window-function columns, 21
  columns total). Same joins, same filter, same window-function expressions —
  no column added, renamed, or dropped; verified row counts (32,581) and column
  counts (18 / 21) match the old combined query exactly before deleting it.
- The point: "don't use the window columns as a feature" was previously only a
  comment + a `CLAUDE.md` rule (i.e. relies on remembering to read them).
  `ml_features` now structurally cannot expose those 3 columns — training code
  querying it gets 18 columns, full stop. `CLAUDE.md`'s existing rule is now a
  backup reminder, not the only safeguard.
- `notebooks/01_eda.ipynb` updated to `pd.read_sql("SELECT * FROM ml_features")`
  instead of reading the .sql file from disk — re-executed clean, `df.shape`
  correctly dropped from `(32581, 21)` to `(32581, 18)`, 0 errors, no downstream
  cell depended on the removed columns (confirmed before deleting the old file).
- `app/app.py` doesn't reference `feature_engineering.sql` or duplicate the
  window functions inline yet (session 13-16 UI is still just TODO placeholders)
  — nothing to migrate there now, but when that Streamlit dashboard tab gets
  built, it should read `dashboard_aggregates`, not `ml_features`.
- Re-ran `scripts/dump_db.sh` after creating the views, since `db-seed/01_seed.sql`
  is a full `pg_dump` snapshot (schema + data + view definitions) taken *before*
  `views.sql` existed — an un-refreshed seed would restore the 4 tables but not
  the views. Verified end-to-end: `docker compose down -v && up -d` now restores
  both views along with the data, no manual `psql -f sql/views.sql` needed.
- `docs/Credit_Risk_Pipeline_Plan_v3.md`'s buổi 9 prompt referenced
  `sql/feature_engineering.sql` by name and said to manually exclude the 3
  window columns — updated to reference `ml_features` instead, since the
  exclusion is now automatic. Buổi 6-7's text describing writing "1 câu SQL"
  was left as-is (historical record of what was actually asked/built then);
  the architecture evolving afterward doesn't rewrite what that session did.

## 2026-09-10 — session 9 scaffold: ml_features column gap fixed, model/ created

- `ml_features` (created in the PR #4 view split) was missing 6 real, 100%-populated
  columns that buổi 9's own prompt requires for `OneHotEncoder`/features: `gender`,
  `marital_status`, `education_level`, `employment_type` (`customers`), `open_accounts`
  (`credit_bureau`), plus `other_debt`, `loan_term_months` (`loans`). Traced back to the
  original buổi 6-7 CTE, which never selected them — carried through unchanged by the
  view split. Verified via `count(*) FILTER (WHERE col IS NOT NULL)` that all 7 are
  fully populated (32,581/32,581) before adding them, so this is a real gap, not an
  intentional omission.
- Fixing it hit a real Postgres constraint: `CREATE OR REPLACE VIEW` can only *append*
  columns at the end, not reorder/insert them — reordering the `SELECT` list raises
  `cannot change name of view column`. New columns were appended after the original
  buổi 6-7 columns instead of interleaved. `dashboard_aggregates` (`SELECT * FROM
  ml_features` + window functions) needed a `DROP VIEW` + recreate, not just `CREATE OR
  REPLACE`, since the `*` expansion shifted its column positions too. Documented both
  gotchas as comments in `sql/views.sql` for next time a column gets added.
  `ml_features` is now 25 columns (was 18), `dashboard_aggregates` 28 (was 21); row
  counts unchanged (32,581). Re-verified portability with a full
  `docker compose down -v && up -d` cycle, then refreshed `db-seed/01_seed.sql`.
- Created `model/features.py` (column bookkeeping: `ID_COLS`, `TARGET_COL`,
  `LEAKAGE_COLS`, `NOMINAL_CATEGORICAL_COLS`, `get_feature_columns`,
  `split_numeric_categorical`) and `model/preprocessing.py` — the latter's
  `build_preprocessor()`/`build_pipeline()` are deliberately left as
  `NotImplementedError` stubs. Buổi 9's own text says to write the
  `ColumnTransformer`/`Pipeline` by hand, not have the AI agent generate it, so the
  interview-explainability goal isn't hollowed out — same spirit as buổi 8's
  self-derived leakage analysis. Scaffolding (data load, feature-set split, stratified
  train/test split) was written normally since it's plumbing, not the graded exercise.
- `notebooks/02_baseline_model.ipynb` created and executed up to (and confirmed
  stopping exactly at) the `NotImplementedError` in the pipeline-training cell —
  everything before it (DB load via `ml_features`, `df.shape`, feature column lists,
  stratified split) runs clean. No ROC-AUC/PR-AUC numbers exist yet; those depend on
  the user's own `build_pipeline()` implementation.

## 2026-09-12 — PR hygiene rules added to CLAUDE.md

- `CLAUDE (1).md` appeared untracked in the repo root — turned out to be the
  contributor guide for "Superpowers," an unrelated Claude Code skills plugin (PR
  template, `dev`-branch targeting, skill eval harness, its own branding). Almost none
  of it applies to a solo portfolio project with no plugin/skill code. Left the file
  alone (untracked, user's to deal with) rather than deleting it unasked.
- Extracted 3 genuinely generalizable habits into `CLAUDE.md`'s Git workflow section:
  disclose AI authorship in commits/PRs (already de facto practice via the
  `Co-Authored-By` trailer, now made an explicit rule), don't bundle unrelated changes
  into one PR (already followed once, in PR #6 splitting off from #5), verify before
  claiming a fix is done (this repo's whole existing pattern of confirming against a
  live DB/pytest/notebook run — now stated as an explicit rule instead of only
  demonstrated ad hoc).

## 2026-09-12 — fixed CLAUDE.md's own documented psql command (it was broken)

- `CLAUDE.md`'s Commands section documented `docker exec -i credit-db psql -U postgres
  -d credit_db -f sql/schema.sql` for rebuilding from scratch. Verified directly:
  `psql: error: sql/schema.sql: No such file or directory`. `-f <path>` with `docker
  exec` looks for that path **inside the container**, which only has `./db-seed`
  mounted (`docker-compose.yml`) — the repo's own `sql/` directory was never visible
  there. README.md had copied the same broken form.
- Fixed both to `-f - < sql/schema.sql` (reads from stdin, which the host-side `<`
  redirect actually supplies) and verified it applies cleanly against the live DB.
  `.github/workflows/ci.yml`'s own apply step is unaffected — it runs `psql` directly
  on the runner (no `docker exec`), where the plain path form is correct as-is; the two
  contexts need different syntax and CLAUDE.md now says so.
- This had apparently never been run as written since it was first documented — a
  reminder that a command sitting in a doc file is a claim, not a fact, until it's
  actually executed.

## Open questions — not yet resolved

- `income` (max ~6,000,000) and `other_debt` (max ~1,190,000) have heavy right tails. Not yet determined whether these are genuine high earners or data errors — currently left uncapped. If model calibration looks off in the tails during buổi 9-11, revisit this before assuming the model is at fault.
