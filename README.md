# Credit Risk & Loan Approval Pipeline

Scaffold cho kế hoạch 17 buổi trong `Credit_Risk_Pipeline_Plan.md`, đã tích hợp sẵn các sửa lỗi
phát hiện được khi audit trực tiếp `Credit_Risk_Dataset.xlsx` (32,581 dòng, 29 cột).

## Khác biệt so với plan gốc — đọc trước khi bắt đầu buổi 1

| # | Vấn đề | Vị trí sửa |
|---|---|---|
| 1 | `loan_grade`/`loan_int_rate` tương quan gần tất định với target → leakage nếu dùng làm input cho form duyệt hồ sơ mới | `app/app.py`, tách 2 model bundle |
| 2 | 5 dòng `person_age`=144/123, 2 dòng `person_emp_length`=123 ở tuổi 21-22 | `etl/historical_load.py::clean_outliers`, `sql/schema.sql` CHECK constraints |
| 3 | `loan_percent_income` và `loan_to_income_ratio` trùng nhau (corr=0.9989) | `etl/historical_load.py::drop_redundant_columns` |
| 4 | Dữ liệu gốc không có cột ngày → dashboard theo tháng (buổi 15) không đủ dữ liệu | `etl/historical_load.py::backfill_loan_dates` |
| 5 | `ON CONFLICT (loan_id)` không hoạt động vì `loan_id` là SERIAL | `sql/schema.sql` (`application_ref`), `etl/daily_ingest.py` |
| 6 | Hồ sơ synthetic hằng ngày lẫn vào tập train nếu không lọc | `sql/schema.sql` (`data_source`), `sql/feature_engineering.sql` |
| 7 | Bảng `locations` lặp toạ độ 32,581 lần dù chỉ có 18 tổ hợp city thật | `sql/schema.sql` (bảng `cities` riêng) |

## Cấu trúc

```
credit-risk-pipeline/
├── sql/
│   ├── schema.sql              # DDL đã sửa — đọc phần comment trong file
│   └── feature_engineering.sql # CTE + window functions (buổi 6-7)
├── etl/
│   ├── config.py                # kết nối Postgres từ .env
│   ├── historical_load.py       # buổi 3-4
│   └── daily_ingest.py          # buổi 5
├── notebooks/
│   └── 01_eda.ipynb             # buổi 8, có checklist sẵn
├── models/                      # buổi 12: model bundle (.pkl) lưu ở đây
├── app/
│   ├── app.py                   # buổi 13-16, Streamlit 2 tab
│   └── utils.py
├── tests/
│   └── test_placeholder.py      # test cho các hàm ETL quan trọng
├── docker-compose.yml           # Postgres cho buổi 1
├── requirements.txt
└── .env.example
```

## Chạy từ đầu

```bash
cp .env.example .env               # sửa mật khẩu thật
docker compose up -d               # buổi 1
psql -h localhost -U postgres -d credit_db -f sql/schema.sql   # buổi 2
python etl/historical_load.py      # buổi 3-4 (sau khi viết split_into_tables)
python etl/daily_ingest.py         # buổi 5 (sau khi viết sample_new_applications)
jupyter notebook notebooks/01_eda.ipynb   # buổi 8
streamlit run app/app.py           # buổi 13-16
```

## Nguyên tắc làm việc với AI agent

Giữ nguyên nguyên tắc từ plan gốc: mỗi buổi đúng 1 prompt, chạy thử và kiểm tra trước khi
sang buổi tiếp theo. Các hàm còn `TODO`/`NotImplementedError` trong scaffold này chính là chỗ
để dán prompt của từng buổi vào.
