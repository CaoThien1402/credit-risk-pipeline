# Credit Risk & Loan Approval Pipeline — Kế hoạch thực thi (v2, đã sửa)

**Dataset**: `Credit_Risk_Dataset.xlsx` — 32,581 hồ sơ vay thật, 29 cột, 3 quốc gia (Canada/UK/USA), tỷ lệ default 21.8%, missing thật ở `person_emp_length` (895 dòng) và `loan_int_rate` (3,116 dòng).

**Nhịp độ**: 1 buổi/ngày → tổng **17 buổi** (~3.5 tuần nếu làm liên tục các ngày trong tuần) — không đổi so với bản gốc.

**So với bản v1**: đã audit trực tiếp file Excel và phát hiện 7 vấn đề (2 dòng outlier tuổi vô lý, leakage nghiêm trọng từ `loan_grade`/`loan_int_rate`, 2 cột trùng lặp, thiếu cột ngày, xung đột SERIAL PK với UPSERT, lẫn dữ liệu synthetic vào tập train, bảng `locations` chưa chuẩn hoá). Các buổi bị ảnh hưởng được đánh dấu 🔧.

---

## Schema cơ sở dữ liệu (PostgreSQL) — đã sửa

Tách file phẳng 29 cột thành **5 bảng** chuẩn hoá (thêm `cities`, bớt `locations`):

```
cities
  city_id (PK), country, state, city, latitude, longitude
  UNIQUE(country, state, city)

customers
  client_id (PK), age, income, home_ownership, emp_length,
  gender, marital_status, education_level, employment_type,
  city_id (FK -> cities)

credit_bureau
  client_id (PK, FK -> customers), default_on_file, cred_hist_length,
  open_accounts, credit_utilization_ratio, past_delinquencies

loans
  loan_id (PK, SERIAL), application_ref (UNIQUE, dùng cho UPSERT),
  client_id (FK -> customers), loan_intent, loan_grade,
  loan_amnt, loan_int_rate, loan_term_months, loan_status (cho phép NULL),
  loan_percent_income, other_debt, debt_to_income_ratio,
  loan_date, data_source ('historical' | 'synthetic_daily')
```

*Bỏ*: bảng `locations` riêng (gộp vào `customers.city_id`) và cột `loan_to_income_ratio` (trùng >99.8% với `loan_percent_income`).
*Thêm*: `cities`, `application_ref`, `loan_date`, `data_source`.

---

## Buổi 1 — Setup môi trường

- Cài PostgreSQL (hoặc Docker: `docker run --name credit-db -e POSTGRES_PASSWORD=xxx -p 5432:5432 -d postgres`)
- Cài pgAdmin hoặc DBeaver
- Tạo virtualenv, cài: `psycopg2-binary sqlalchemy pandas scikit-learn xgboost imbalanced-learn shap streamlit plotly openpyxl joblib python-dotenv`
- Tạo cấu trúc repo: `/etl`, `/sql`, `/notebooks`, `/app`, `/models`, `/tests`, `README.md`

**Kết quả buổi**: Postgres chạy được, kết nối test thành công từ Python.

---

## Buổi 2 — DDL tạo bảng 🔧

**Prompt AI**:
> "Viết DDL PostgreSQL tạo 5 bảng: cities, customers, credit_bureau, loans theo schema sau [dán schema ở trên]. customers.client_id là PK dạng VARCHAR, city_id là FK về cities. loans.loan_id tự sinh SERIAL nhưng thêm cột application_ref VARCHAR UNIQUE dùng làm khoá cho UPSERT (vì loan_id SERIAL không dùng được cho ON CONFLICT khi dữ liệu được sinh mới mỗi lần chạy). loans.loan_status cho phép NULL. Thêm CHECK constraint: customers.age BETWEEN 18 AND 100, customers.emp_length <= age - 14, loans.loan_status IN (0,1), loans.data_source IN ('historical','synthetic_daily')."

- Chạy DDL, kiểm tra 5 bảng đã tạo đúng bằng pgAdmin/DBeaver, thử insert 1 dòng age=150 để xác nhận CHECK constraint chặn được.

**Kết quả buổi**: 5 bảng rỗng, có khóa chính/khóa ngoại và CHECK constraint đúng.

---

## Buổi 3-4 — ETL nạp dữ liệu thật (historical bulk load) 🔧

**Prompt AI**:
> "Viết script Python dùng pandas + SQLAlchemy đọc file Credit_Risk_Dataset.xlsx, tách cột theo schema 5 bảng [dán schema]. Trước khi insert: (1) set NaN cho person_age > 100 và cho person_emp_length > (person_age - 14) — đây là lỗi dữ liệu thật cần chặn trước khi impute; (2) median imputation cho person_emp_length và loan_int_rate SAU bước chặn outlier; (3) bỏ cột loan_to_income_ratio (trùng loan_percent_income); (4) sinh loan_date rải đều ngẫu nhiên trong 24 tháng gần nhất và application_ref = uuid4 cho mỗi dòng, gắn data_source='historical'. Ghi vào PostgreSQL theo transaction, insert cities trước để lấy city_id."

- Buổi 3: viết script, insert thử với 1000 dòng đầu, kiểm tra dữ liệu vào đúng bảng, xác nhận không còn `person_age > 100`.
- Buổi 4: chạy full 32,581 dòng, viết query kiểm tra đếm số dòng mỗi bảng khớp nhau, xác nhận `loan_date` rải đều qua 24 tháng và `data_source='historical'` toàn bộ.

**Kết quả buổi**: toàn bộ 32,581 hồ sơ nằm trong Postgres, phân bổ đúng 5 bảng, không còn outlier tuổi/thâm niên vô lý.

---

## Buổi 5 — Mô phỏng luồng hồ sơ vay mới hằng ngày 🔧

**Prompt AI**:
> "Viết script Python sinh 50-100 hồ sơ vay mới mỗi ngày bằng cách lấy mẫu có nhiễu từ phân phối thống kê (mean/std/tần suất) của bảng loans và customers HIỆN CÓ TRONG POSTGRES nhưng CHỈ tính trên WHERE data_source='historical' (không lấy nhiễu từ dữ liệu synthetic ngày hôm trước, tránh phân phối trôi dần). Mỗi hồ sơ mới có application_ref = hash(client_id + ngày hôm nay), loan_date = ngày hôm nay, data_source='synthetic_daily', loan_status=NULL (hồ sơ đang chờ duyệt, chưa có outcome). Insert với ON CONFLICT (application_ref) DO UPDATE."

- Chạy script 2 lần liên tiếp trong cùng ngày, xác nhận số dòng KHÔNG tăng gấp đôi (test UPSERT thật, không phải giả vì id luôn mới như ở loan_id).
- Cài Cron (Linux) hoặc Task Scheduler (Windows) chạy script này lúc 20:00 mỗi ngày.

**Kết quả buổi**: pipeline daily ingestion tự động, idempotent thật theo `application_ref`.

---

## Buổi 6-7 — SQL nâng cao: Feature Engineering 🔧

**Prompt AI**:
> "Viết 1 câu SQL PostgreSQL dùng CTE nối 5 bảng cities, customers, credit_bureau, loans; **bắt buộc thêm WHERE data_source = 'historical'** trong CTE gốc để loại hồ sơ synthetic_daily chưa có outcome thật; thêm cột window function tính loan_amnt trung bình theo nhóm tuổi (age bucket 10 năm) và tỷ lệ default trung bình theo loan_grade."

- Buổi 6: viết CTE nối bảng, xác nhận số dòng ra đúng bằng số hồ sơ historical (không lẫn synthetic).
- Buổi 7: thêm window functions (avg theo age bucket, tỷ lệ default theo grade, rank theo credit_utilization_ratio trong từng quốc gia), test kết quả bằng tay trên vài dòng mẫu.

**Kết quả buổi**: 1 câu SQL trả về bảng feature đầy đủ, chỉ chứa dữ liệu lịch sử có nhãn thật, sẵn sàng đọc vào Python bằng SQLAlchemy.

---

## Buổi 8 — EDA & xử lý missing 🔧

- Đọc dữ liệu từ câu SQL ở buổi 6-7 vào pandas.
- Vẽ phân phối `loan_status`, phân tích pattern missing của `loan_int_rate` (missing ngẫu nhiên hay có quy luật theo `loan_grade`?).
- **Kiểm tra tương quan `loan_grade` và `loan_int_rate` với `loan_status`** — nếu default rate chênh lệch cực mạnh giữa các grade (ví dụ A ~10% vs G ~98%), đây là dấu hiệu leakage: 2 cột này là *kết quả* của một quy trình chấm điểm rủi ro, không phải input độc lập tại thời điểm nộp hồ sơ. Ghi rõ quyết định (loại khỏi model duyệt hồ sơ mới) để dùng ở buổi 9.
- Xác nhận lại không còn `person_age`/`person_emp_length` bất thường (đã chặn ở ETL buổi 3-4, đây là bước kiểm tra chéo).

**Kết quả buổi**: dataset sạch, có quyết định rõ ràng về feature set cho buổi 9-11, vài biểu đồ EDA lưu lại cho README.

---

## Buổi 9 — Baseline model 🔧

**Prompt AI**:
> "Viết 2 pipeline scikit-learn Logistic Regression với class_weight='balanced', train/test split stratify theo loan_status: (a) *portfolio* — dùng đầy đủ feature kể cả loan_grade, loan_int_rate; (b) *at-application* — loại bỏ loan_grade và loan_int_rate, chỉ dùng thông tin thô của khách hàng và credit_bureau. Đánh giá cả 2 bằng ROC-AUC và Precision-Recall AUC, in bảng so sánh chênh lệch."

**Kết quả buổi**: 2 baseline, thấy rõ bằng số liệu mức độ leakage giữa 2 feature set — dùng để quyết định model nào đưa vào Streamlit ở buổi 13.

---

## Buổi 10-11 — Model chính: XGBoost 🔧

**Prompt AI**:
> "Viết pipeline scikit-learn: train XGBoost cho cả 2 feature set (portfolio và at-application) với scale_pos_weight tính từ tỷ lệ class (không dùng SMOTE trước), đánh giá bằng ROC-AUC và PR-AUC, so sánh với 2 baseline Logistic Regression tương ứng."

- Buổi 10: train XGBoost với `scale_pos_weight` cho cả 2 bản, so sánh với baseline.
- Buổi 11: thử thêm SMOTE (imbalanced-learn) cho bản *at-application* (bản thực sự dùng trong Streamlit), so sánh 2 cách xử lý imbalance, chọn cách tốt hơn dựa trên PR-AUC.

**Kết quả buổi**: model tốt nhất đã chọn cho từng mục đích, có bảng so sánh baseline vs XGBoost vs XGBoost+SMOTE, tách rõ theo feature set.

---

## Buổi 12 — Explainability (SHAP) 🔧

**Prompt AI**:
> "Viết code dùng thư viện shap để vẽ summary plot cho model XGBoost *at-application* đã train, và force plot cho 1 hồ sơ cụ thể để giải thích quyết định duyệt/từ chối."

- Xác định 3-5 feature quan trọng nhất theo SHAP (bản at-application).
- Lưu model dạng **bundle**, không chỉ model trần: `joblib.dump({"model": model, "preprocessor": preprocessor, "feature_names": features, "trained_at": ..., "metrics": {...}}, 'models/at_application_model.pkl')` — lặp lại tương tự cho `models/portfolio_risk_model.pkl`. Nếu chỉ lưu model trần, Streamlit ở buổi 13 sẽ không tái tạo đúng bước tiền xử lý.

**Kết quả buổi**: 2 file `.pkl` (bundle đầy đủ), có SHAP plots minh họa lý do quyết định.

---

## Buổi 13-14 — Streamlit: màn hình dự đoán 🔧

**Prompt AI**:
> "Viết Streamlit app: form nhập input (tuổi, thu nhập, số tiền vay, mục đích vay, home_ownership, emp_length, các trường credit_bureau, ...) cho nhân viên tín dụng — **KHÔNG có trường loan_grade hay loan_int_rate**, vì đây là bản model at-application. Load `models/at_application_model.pkl` để dự đoán, hiển thị kết quả Duyệt/Từ chối kèm xác suất rủi ro, và top 3 lý do ảnh hưởng đến quyết định dựa trên SHAP values."

- Buổi 13: dựng form input + gọi model, test với vài trường hợp.
- Buổi 14: thêm hiển thị SHAP force plot cho từng dự đoán, chỉnh giao diện.

**Kết quả buổi**: màn hình dự đoán hoàn chỉnh, không rò rỉ thông tin downstream vào input, chạy được `streamlit run app.py`.

---

## Buổi 15-16 — Streamlit: dashboard quản lý 🔧

**Prompt AI**:
> "Viết tab thứ 2 trong Streamlit app: dashboard Plotly đọc trực tiếp từ PostgreSQL, gồm biểu đồ tổng dư nợ giải ngân theo tháng (dùng cột loan_date, chú thích rõ đây là dữ liệu lịch sử backfill + luồng synthetic_daily thật từ ngày triển khai), tỷ lệ default theo country và theo loan_grade (chỉ tính trên data_source='historical', vì synthetic_daily chưa có outcome), và bản đồ scatter_geo join qua customers.city_id -> cities.latitude/longitude."

- Buổi 15: dựng các biểu đồ chính (tổng dư nợ theo tháng, tỷ lệ nợ xấu theo grade/country).
- Buổi 16: thêm bản đồ scatter_geo, tinh chỉnh layout 2 tab.

**Kết quả buổi**: app Streamlit hoàn chỉnh 2 màn hình, chạy ổn định, biểu đồ theo tháng có đủ dữ liệu nhờ backfill ở buổi 3-4.

---

## Buổi 17 — Hoàn thiện & README 🔧

- Viết README: mô tả bài toán, kiến trúc hệ thống (Excel → Postgres → SQL feature engineering → 2 model → Streamlit), **giải thích rõ vì sao có 2 model (at-application vs portfolio) và leakage của loan_grade/loan_int_rate**, hướng dẫn chạy lại từ đầu.
- Chụp screenshot/quay video ngắn demo app, gắn vào README.
- Push code lên Github, tách rõ `/etl`, `/sql`, `/notebooks`, `/app`, `/models`, `/tests`.
- Viết bảng so sánh model (Logistic vs XGBoost vs XGBoost+SMOTE, **× 2 feature set**) kèm lý do chọn PR-AUC làm metric chính và lý do loại loan_grade/loan_int_rate khỏi bản dùng thật.

**Kết quả buổi**: project hoàn chỉnh, sẵn sàng đưa vào CV/portfolio, có luận điểm rõ ràng về leakage — điểm cộng lớn khi phỏng vấn vì đây là lỗi rất dễ mắc và rất hay bị hỏi.

---

## Nguyên tắc làm việc với AI agent

Mỗi buổi chỉ đưa **đúng 1 prompt** (đã ghi sẵn ở từng buổi) cho AI agent, chạy thử và kiểm tra kết quả trước khi copy sang buổi tiếp theo. Không gộp nhiều buổi vào 1 prompt — dễ ra code rối, khó debug.

Các buổi đánh dấu 🔧 có prompt dài hơn bản gốc vì đã gộp sẵn yêu cầu sửa lỗi — dán nguyên văn, đừng rút gọn lại các mệnh đề phụ (cap outlier, backfill ngày, application_ref, lọc data_source, tách feature set): mỗi mệnh đề tương ứng với 1 trong 7 lỗi đã kiểm chứng trên dữ liệu thật.
