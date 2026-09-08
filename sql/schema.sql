-- Credit Risk Pipeline — Schema đã hiệu chỉnh (v2)
-- So với schema gốc trong kế hoạch, các thay đổi được chú thích ngay dưới từng bảng.

CREATE TABLE cities (
    city_id     SERIAL PRIMARY KEY,
    country     VARCHAR(50) NOT NULL,
    state       VARCHAR(50) NOT NULL,
    city        VARCHAR(50) NOT NULL,
    latitude    NUMERIC(9,6) NOT NULL,
    longitude   NUMERIC(9,6) NOT NULL,
    UNIQUE (country, state, city)
);
-- SỬA: dữ liệu thật chỉ có đúng 18 tổ hợp (country, state, city), mỗi city có lat/long cố định 1:1.
-- Tách bảng riêng thay vì lặp lại toạ độ ở mỗi khách hàng (bảng "locations" trong plan gốc) —
-- giảm 32,581 dòng dư thừa xuống 18 dòng, đúng chuẩn 3NF, join scatter_geo (buổi 16) cũng gọn hơn.

CREATE TABLE customers (
    client_id       VARCHAR(20) PRIMARY KEY,
    age             SMALLINT NOT NULL CHECK (age BETWEEN 18 AND 100),
    income          NUMERIC(14,2) NOT NULL CHECK (income >= 0),
    home_ownership  VARCHAR(20) NOT NULL CHECK (home_ownership IN ('RENT','OWN','MORTGAGE','OTHER')),
    emp_length      NUMERIC(5,2) CHECK (emp_length IS NULL OR (emp_length >= 0 AND emp_length <= age - 14)),
    gender          VARCHAR(10),
    marital_status  VARCHAR(20),
    education_level VARCHAR(20),
    employment_type VARCHAR(20),
    city_id         INT REFERENCES cities(city_id),
    created_at      TIMESTAMP NOT NULL DEFAULT now()
);
-- SỬA: CHECK age BETWEEN 18 AND 100 — dữ liệu thật có 5 dòng age = 144/123 (lỗi nhập liệu kinh điển
-- của bộ dataset này). Chặn ngay ở DB thay vì đợi đến EDA (buổi 8) mới tình cờ phát hiện.
-- SỬA: CHECK emp_length <= age - 14 — dữ liệu thật có 2 dòng emp_length = 123 năm trong khi
-- age chỉ 21-22 (vô lý về mặt logic, không chỉ là "hiếm").
-- SỬA: city_id thay cho bảng "locations" tách riêng — quan hệ khách hàng-thành phố là 1:1
-- tại một thời điểm, không cần bảng con (chỉ tách riêng nếu sau này track lịch sử đổi địa chỉ).

CREATE TABLE credit_bureau (
    client_id                 VARCHAR(20) PRIMARY KEY REFERENCES customers(client_id),
    default_on_file           CHAR(1) NOT NULL CHECK (default_on_file IN ('Y','N')),
    cred_hist_length          SMALLINT NOT NULL CHECK (cred_hist_length >= 0),
    open_accounts             SMALLINT CHECK (open_accounts >= 0),
    credit_utilization_ratio  NUMERIC(5,4) CHECK (credit_utilization_ratio BETWEEN 0 AND 1),
    past_delinquencies        SMALLINT CHECK (past_delinquencies >= 0)
);

CREATE TABLE loans (
    loan_id             SERIAL PRIMARY KEY,
    application_ref     VARCHAR(64) NOT NULL UNIQUE,
    client_id           VARCHAR(20) NOT NULL REFERENCES customers(client_id),
    loan_intent         VARCHAR(30),
    loan_grade          CHAR(1) CHECK (loan_grade IN ('A','B','C','D','E','F','G')),
    loan_amnt           NUMERIC(14,2) NOT NULL CHECK (loan_amnt > 0),
    loan_int_rate       NUMERIC(5,2),
    loan_term_months    SMALLINT CHECK (loan_term_months IN (12,24,36,60)),
    loan_status         SMALLINT CHECK (loan_status IN (0,1)),  -- NULL = hồ sơ synthetic đang chờ model dự đoán
    loan_percent_income NUMERIC(6,4),
    other_debt          NUMERIC(14,2),
    debt_to_income_ratio NUMERIC(6,4),
    loan_date           DATE NOT NULL,
    data_source         VARCHAR(16) NOT NULL DEFAULT 'historical'
                         CHECK (data_source IN ('historical','synthetic_daily')),
    created_at          TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX idx_loans_client ON loans(client_id);
CREATE INDEX idx_loans_date   ON loans(loan_date);
CREATE INDEX idx_loans_source ON loans(data_source);

-- SỬA (những thay đổi quan trọng nhất so với plan gốc):
--
-- 1) application_ref (UNIQUE, tự sinh ở tầng ETL — vd uuid4 hoặc hash(client_id + loan_date))
--    dùng làm khoá cho "ON CONFLICT ... DO UPDATE" ở buổi 5. loan_id SERIAL KHÔNG dùng được
--    cho UPSERT vì mỗi lần script sinh dữ liệu mới, Postgres luôn cấp id mới -> không bao giờ
--    conflict thật, script "test idempotent" trong plan gốc sẽ luôn pass giả tạo.
--
-- 2) Bỏ cột loan_to_income_ratio khỏi bảng loans (chỉ giữ loan_percent_income) — 2 cột này
--    corr = 0.9989 trên dữ liệu thật, tức cùng một tín hiệu tính 2 lần. Giữ cả hai gây
--    multicollinearity thừa và làm SHAP chia đôi importance của cùng một biến.
--
-- 3) Thêm loan_date + data_source — dữ liệu gốc KHÔNG có bất kỳ cột ngày/giờ nào. Nếu không
--    backfill, biểu đồ "tổng dư nợ theo tháng" (buổi 15) chỉ có dữ liệu của vài ngày synthetic,
--    không đủ để vẽ xu hướng theo tháng có ý nghĩa.
--    -> Script ETL lịch sử (buổi 3-4) cần rải loan_date giả lập đều trong ~24 tháng gần nhất.
--
-- 4) data_source dùng để lọc "WHERE data_source = 'historical'" khi train model (buổi 9-11) —
--    tránh học nhầm loan_status giả lập (sampled từ phân phối, không phải outcome thật) của các
--    hồ sơ synthetic được chèn liên tục từ buổi 5 trở đi.
--
-- 5) loan_status cho phép NULL: hồ sơ synthetic mới sinh mô phỏng đúng thực tế — một đơn vay
--    mới nộp CHƯA có kết quả default/không default, đó chính là cái model phải dự đoán.

-- GHI CHÚ VỀ FEATURE LEAKAGE (áp dụng ở buổi 9-11 khi build model, không thuộc phạm vi DDL):
-- loan_grade và loan_int_rate tương quan gần tất định với target (xem phân tích trong hội thoại).
-- Model "duyệt/từ chối hồ sơ mới" (buổi 13) KHÔNG được dùng 2 cột này làm input — chỉ dùng thông
-- tin thô (customers + credit_bureau). Chỉ model "portfolio risk" (phân tích danh mục đã có) mới
-- được dùng đủ cột, và phải ghi rõ sự khác biệt này trong README để tránh overclaim.
