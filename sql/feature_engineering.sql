-- Buổi 6-7: CTE nối 4 bảng + window functions.
-- SỬA quan trọng: WHERE l.data_source = 'historical' — chỉ dùng hồ sơ có outcome thật để
-- train model, không lẫn loan_status giả lập (NULL) của hồ sơ synthetic_daily.

WITH base AS (
    SELECT
        l.loan_id,
        l.client_id,
        l.loan_intent,
        l.loan_grade,
        l.loan_amnt,
        l.loan_int_rate,
        l.loan_percent_income,
        l.debt_to_income_ratio,
        l.loan_status,
        c.age,
        c.income,
        c.home_ownership,
        c.emp_length,
        cb.default_on_file,
        cb.cred_hist_length,
        cb.credit_utilization_ratio,
        cb.past_delinquencies,
        ci.country
    FROM loans l
    JOIN customers c      ON c.client_id = l.client_id
    JOIN credit_bureau cb ON cb.client_id = l.client_id
    LEFT JOIN cities ci   ON ci.city_id = c.city_id
    WHERE l.data_source = 'historical'   -- <<< dòng sửa quan trọng nhất trong file này
)
SELECT
    base.*,
    -- TODO (buổi 7): window functions
    AVG(loan_amnt) OVER (PARTITION BY (age / 10) * 10) AS avg_loan_amnt_by_age_bucket,
    AVG(loan_status::numeric) OVER (PARTITION BY loan_grade) AS default_rate_by_grade,
    RANK() OVER (PARTITION BY country ORDER BY credit_utilization_ratio DESC) AS util_rank_in_country
FROM base;
