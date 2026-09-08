"""Simulate 50-100 new loan applications per day and upsert them into Postgres."""
import sys
import hashlib
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import text

from config import get_engine

# Numeric columns: Gaussian-sampled around the historical mean/std, clipped to valid ranges.
NUMERIC_COLS = ["loan_amnt", "loan_int_rate", "loan_percent_income", "other_debt", "debt_to_income_ratio"]
NUMERIC_MIN = {
    "loan_amnt": 1.0,               # CHECK loan_amnt > 0
    "loan_int_rate": 0.0,
    "loan_percent_income": 0.0,
    "other_debt": 0.0,
    "debt_to_income_ratio": 0.0,
}
NUMERIC_DECIMALS = {
    "loan_amnt": 2,                 # NUMERIC(14,2)
    "loan_int_rate": 2,             # NUMERIC(5,2)
    "loan_percent_income": 4,       # NUMERIC(6,4)
    "other_debt": 2,                # NUMERIC(14,2)
    "debt_to_income_ratio": 4,      # NUMERIC(6,4)
}

# Categorical columns: sampled by historical frequency, always valid per the CHECK
# constraints (loan_grade A-G, loan_term_months 12/24/36/60).
CATEGORICAL_COLS = ["loan_intent", "loan_grade", "loan_term_months"]


def load_reference_stats() -> pd.DataFrame:
    """Read historical loans as the statistical baseline for sampling synthetic applications."""
    # Filtered to data_source='historical' only, so synthetic noise never compounds across days.
    engine = get_engine()
    query = """
        SELECT client_id, loan_intent, loan_grade, loan_amnt, loan_int_rate,
               loan_term_months, loan_percent_income, other_debt, debt_to_income_ratio
        FROM loans
        WHERE data_source = 'historical'
    """
    return pd.read_sql(query, engine)


def sample_new_applications(n: int, reference_stats: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
    """Sample n synthetic loan applications with noise drawn from reference_stats."""
    rng = np.random.default_rng(seed)

    client_ids = rng.choice(reference_stats["client_id"].to_numpy(), size=n, replace=True)
    data = {"client_id": client_ids}

    for col in NUMERIC_COLS:
        mean = reference_stats[col].mean()
        std = reference_stats[col].std()
        sampled = rng.normal(loc=mean, scale=std, size=n)
        sampled = np.clip(sampled, NUMERIC_MIN[col], None)
        data[col] = np.round(sampled, NUMERIC_DECIMALS[col])

    for col in CATEGORICAL_COLS:
        freq = reference_stats[col].value_counts(normalize=True)
        data[col] = rng.choice(freq.index.to_numpy(), size=n, p=freq.to_numpy())

    df = pd.DataFrame(data)
    df["loan_term_months"] = df["loan_term_months"].astype(int)
    return df


def make_application_ref(client_id: str, application_date: date) -> str:
    raw = f"{client_id}-{application_date.isoformat()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def upsert_loans(df: pd.DataFrame) -> None:
    engine = get_engine()
    today = date.today()
    df = df.copy()
    # Keyed on application_ref (client_id + date hash), not loan_id — loan_id is a SERIAL
    # and would never match on conflict.
    df["application_ref"] = [make_application_ref(cid, today) for cid in df["client_id"]]
    df["loan_date"] = today
    df["data_source"] = "synthetic_daily"
    df["loan_status"] = None  # pending model prediction, not a real label

    columns = list(df.columns)
    insert_cols = ", ".join(columns)
    value_placeholders = ", ".join(f":{c}" for c in columns)
    update_cols = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns if c != "application_ref")

    stmt = text(f"""
        INSERT INTO loans ({insert_cols})
        VALUES ({value_placeholders})
        ON CONFLICT (application_ref) DO UPDATE SET {update_cols}
    """)
    with engine.begin() as conn:
        conn.execute(stmt, df.to_dict(orient="records"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    reference_stats = load_reference_stats()
    n = int(np.random.randint(50, 101))
    new_apps = sample_new_applications(n, reference_stats)
    upsert_loans(new_apps)
    print(f"Upserted {len(new_apps)} synthetic applications for {date.today()}.")
