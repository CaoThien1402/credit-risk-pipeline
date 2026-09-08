"""Buổi 3-4: nạp toàn bộ Credit_Risk_Dataset.xlsx (32,581 dòng) vào PostgreSQL.

Đây là SKELETON — mỗi hàm có TODO tương ứng với 1 prompt AI trong kế hoạch gốc.
Đã có sẵn phần xử lý outlier/backfill ngày vì đây là lỗi dữ liệu thật đã kiểm chứng,
không phải phần "tự luyện" của buổi học.
"""
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
    """Chặn lỗi dữ liệu đã kiểm chứng: age > 100 (5 dòng thật là 144/123) và
    emp_length vô lý so với tuổi (2 dòng thật là 123 năm ở tuổi 21-22).
    Cap thay vì xoá để không mất thông tin các cột khác của hồ sơ.
    """
    df = df.copy()
    df.loc[df["person_age"] > 100, "person_age"] = np.nan
    bad_emp = df["person_emp_length"] > (df["person_age"] - 14)
    df.loc[bad_emp, "person_emp_length"] = np.nan
    # TODO (buổi 8): quyết định impute lại age/emp_length bằng median hay KNN,
    # so sánh phân phối trước/sau — đây chỉ mới CHẶN, chưa IMPUTE.
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Median imputation cho person_emp_length, loan_int_rate — theo đúng prompt buổi 3-4.
    Lưu ý: sau clean_outliers ở trên, person_emp_length có thêm vài NaN mới — impute sau
    khi đã cap outlier, không impute trước.

    person_age cũng được median-impute tạm thời tại đây: customers.age là NOT NULL trong
    schema, nên 5 dòng bị clean_outliers set NaN (age gốc 144/123) phải có giá trị mới
    insert được. Đây là quyết định TẠM cho buổi 3-4 để không mất dữ liệu — buổi 8 (EDA)
    có thể xem lại và đổi chiến lược (KNN, impute theo nhóm...) bằng UPDATE nếu cần.
    """
    df = df.copy()
    df["person_age"] = df["person_age"].fillna(df["person_age"].median())
    df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())
    df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())
    return df


def backfill_loan_dates(df: pd.DataFrame, months_back: int = 24, seed: int = 42) -> pd.DataFrame:
    """Dữ liệu gốc KHÔNG có cột ngày. Rải loan_date giả lập đều trong `months_back` tháng gần nhất
    để dashboard "tổng dư nợ theo tháng" (buổi 15) có đủ lịch sử để vẽ trend.
    """
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
    """loan_percent_income và loan_to_income_ratio trùng nhau (corr=0.9989) — giữ 1 cột."""
    return df.drop(columns=["loan_to_income_ratio"], errors="ignore")


def split_into_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Tách file phẳng theo schema 4 bảng trong sql/schema.sql (cities, customers,
    credit_bureau, loans). customers giữ tạm (country, state, city) làm khoá tự nhiên
    để load_to_postgres nối sang city_id sau khi cities đã insert (city_id là SERIAL,
    chỉ biết giá trị thật sau khi ghi vào Postgres).
    """
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
    """Insert full 32,581 dòng theo transaction: cities trước để lấy city_id sinh tự động
    (SERIAL), nối vào customers, rồi mới insert customers/credit_bureau/loans (đều FK về
    customers/cities nên phải theo đúng thứ tự này).
    """
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
    print(f"Đã nạp {len(raw)} hồ sơ vào PostgreSQL.")
