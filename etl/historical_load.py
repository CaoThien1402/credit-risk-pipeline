"""Load Credit_Risk_Dataset.xlsx (32,581 rows) into PostgreSQL."""
import sys
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config import get_engine

SOURCE_FILE = "data/Credit_Risk_Dataset.xlsx"


def load_raw(path: str = SOURCE_FILE) -> pd.DataFrame:
    return pd.read_excel(path)


def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap invalid person_age and person_emp_length outliers to NaN."""
    # Capped, not dropped, so the rest of each row's data survives.
    df = df.copy()
    df.loc[df["person_age"] > 100, "person_age"] = np.nan
    bad_emp = df["person_emp_length"] > (df["person_age"] - 14)
    df.loc[bad_emp, "person_emp_length"] = np.nan
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Median-impute person_age, person_emp_length, and loan_int_rate."""
    # Must run after clean_outliers: capping creates new NaNs that need imputing.
    # person_age is imputed here too because customers.age is NOT NULL in the schema;
    # revisit the strategy (KNN, group median, ...) during EDA if needed.
    df = df.copy()
    df["person_age"] = df["person_age"].fillna(df["person_age"].median())
    df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())
    df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())
    return df


def backfill_loan_dates(df: pd.DataFrame, months_back: int = 24, seed: int = 42) -> pd.DataFrame:
    """Backfill a synthetic loan_date spread evenly across the last `months_back` months."""
    # Source data has no date column at all; this gives monthly trend charts history to show.
    rng = np.random.default_rng(seed)
    end = datetime.today()
    start = end - timedelta(days=months_back * 30)
    offsets = rng.integers(0, (end - start).days, size=len(df))
    df = df.copy()
    df["loan_date"] = [start + timedelta(days=int(o)) for o in offsets]
    df["data_source"] = "historical"
    df["application_ref"] = [str(uuid.uuid4()) for _ in range(len(df))]
    return df


def drop_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop loan_to_income_ratio, a near-duplicate of loan_percent_income."""
    # corr = 0.9989 on the real data — same signal, keep only one.
    return df.drop(columns=["loan_to_income_ratio"], errors="ignore")


def split_into_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the flat dataframe into cities/customers/credit_bureau/loans per schema.sql."""
    # customers keeps (country, state, city) temporarily; load_to_postgres resolves it to
    # city_id once cities are inserted (city_id is a SERIAL, only known after insert).
    cities = (
        df[["country", "state", "city", "city_latitude", "city_longitude"]]
        .drop_duplicates(subset=["country", "state", "city"])
        .rename(columns={"city_latitude": "latitude", "city_longitude": "longitude"})
        .reset_index(drop=True)
    )

    customers = df.rename(
        columns={
            "client_ID": "client_id",
            "person_age": "age",
            "person_income": "income",
            "person_home_ownership": "home_ownership",
            "person_emp_length": "emp_length",
        }
    )[
        [
            "client_id", "age", "income", "home_ownership", "emp_length",
            "gender", "marital_status", "education_level", "employment_type",
            "country", "state", "city",
        ]
    ]

    credit_bureau = df.rename(
        columns={
            "client_ID": "client_id",
            "cb_person_default_on_file": "default_on_file",
            "cb_person_cred_hist_length": "cred_hist_length",
        }
    )[
        [
            "client_id", "default_on_file", "cred_hist_length",
            "open_accounts", "credit_utilization_ratio", "past_delinquencies",
        ]
    ]

    loans = df.rename(columns={"client_ID": "client_id"})[
        [
            "application_ref", "client_id", "loan_intent", "loan_grade", "loan_amnt",
            "loan_int_rate", "loan_term_months", "loan_status", "loan_percent_income",
            "other_debt", "debt_to_income_ratio", "loan_date", "data_source",
        ]
    ]

    return {
        "cities": cities,
        "customers": customers,
        "credit_bureau": credit_bureau,
        "loans": loans,
    }


def load_to_postgres(tables: dict[str, pd.DataFrame]) -> None:
    """Insert all tables into Postgres in one transaction, cities first."""
    # Order matters: customers/credit_bureau/loans all carry FKs back to cities/customers.
    engine = get_engine()
    with engine.begin() as conn:
        tables["cities"].to_sql("cities", conn, if_exists="append", index=False)

        city_lookup = pd.read_sql("SELECT city_id, country, state, city FROM cities", conn)
        customers = tables["customers"].merge(
            city_lookup, on=["country", "state", "city"], how="left"
        ).drop(columns=["country", "state", "city"])
        customers.to_sql("customers", conn, if_exists="append", index=False)

        tables["credit_bureau"].to_sql("credit_bureau", conn, if_exists="append", index=False)
        tables["loans"].to_sql("loans", conn, if_exists="append", index=False)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raw = load_raw()
    raw = clean_outliers(raw)
    raw = impute_missing(raw)
    raw = backfill_loan_dates(raw)
    raw = drop_redundant_columns(raw)
    tables = split_into_tables(raw)
    load_to_postgres(tables)
    print(f"Loaded {len(raw)} records into PostgreSQL.")
