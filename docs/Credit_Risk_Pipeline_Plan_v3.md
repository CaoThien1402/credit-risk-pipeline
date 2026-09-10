# Credit Risk & Loan Approval Pipeline — Kế hoạch thực thi (v3, đã sửa)

**Dataset**: `Credit_Risk_Dataset.xlsx` — 32,581 hồ sơ vay thật, 29 cột, 3 quốc gia (Canada/UK/USA), tỷ lệ default 21.8%, missing thật ở `person_emp_length` (895 dòng) và `loan_int_rate` (3,116 dòng).

**Nhịp độ**: 1 buổi/ngày → tổng **17 buổi** (~3.5 tuần nếu làm liên tục các ngày trong tuần) — không đổi so với bản gốc.

**So với bản v2**: sửa 2 vấn đề mới, khác loại với 7 vấn đề trước (những vấn đề trước là *feature leakage* — cột dữ liệu rò rỉ target vào model; 2 vấn đề này là *spoiler/prompt leakage* — kết quả phân tích đã audit sẵn bị nhét vào chính bước lẽ ra phải tự khám phá):
1. Buổi 8 (EDA) trước đây nêu sẵn con số cụ thể ("A ~10% vs G ~98%") thay vì dạy phương pháp — agent/bạn chỉ cần lặp lại câu đó mà không thực sự tính, buổi khám phá trở nên giả.
2. `CLAUDE.md`/`MEMORY.md` (nếu dùng Claude Code) chứa sẵn facts đã audit — đúng mục đích thiết kế của 2 file này, nhưng nếu có mặt từ buổi 1 thì agent làm buổi 8-9 sẽ "biết trước" đáp án. Dời việc thêm 2 file này sang buổi 10.

Đồng thời tích hợp workflow Docker (pg_dump seeding) vào buổi 1.

---

## Schema cơ sở dữ liệu (PostgreSQL) — không đổi so với v2

Tách file phẳng 29 cột thành **4 bảng** chuẩn hoá (thêm `cities`, bớt `locations`):

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

## Buổi 1 — Setup môi trường 🔧

- Cài Docker, chạy Postgres qua Compose (không dùng lệnh `docker run` rời như bản cũ):
  ```bash
  docker compose up -d
  ```
  `docker-compose.yml` mount `./db-seed` vào `/docker-entrypoint-initdb.d` — bất kỳ file `.sql` nào trong đó tự nạp đúng 1 lần khi data dir rỗng. Chưa cần file gì ở buổi này, nhưng cấu trúc đã sẵn sàng để dùng ở buổi 4.
- Cài pgAdmin hoặc DBeaver.
- Tạo virtualenv, cài: `psycopg2-binary sqlalchemy pandas scikit-learn xgboost imbalanced-learn shap streamlit plotly openpyxl joblib python-dotenv`.
- Tạo cấu trúc repo: `/etl`, `/sql`, `/notebooks`, `/app`, `/models`, `/tests`, `/scripts`, `/db-seed`, `README.md`, `docker-compose.yml`.
- **Chưa tạo `CLAUDE.md`/`MEMORY.md` ở buổi này** — xem lý do và thời điểm đúng trong "Nguyên tắc làm việc với AI agent" ở cuối file.

**Kết quả buổi**: Postgres chạy được qua `docker compose up -d`, kết nối test thành công từ Python.

---

## Buổi 2 — DDL tạo bảng 🔧

**Prompt AI**:
> "Viết DDL PostgreSQL tạo 4 bảng: cities, customers, credit_bureau, loans theo schema sau [dán schema ở trên]. customers.client_id là PK dạng VARCHAR, city_id là FK về cities. loans.loan_id tự sinh SERIAL nhưng thêm cột application_ref VARCHAR UNIQUE dùng làm khoá cho UPSERT (vì loan_id SERIAL không dùng được cho ON CONFLICT khi dữ liệu được sinh mới mỗi lần chạy). loans.loan_status cho phép NULL. Thêm CHECK constraint: customers.age BETWEEN 18 AND 100, customers.emp_length <= age - 14, loans.loan_status IN (0,1), loans.data_source IN ('historical','synthetic_daily')."

- Chạy DDL, kiểm tra 4 bảng đã tạo đúng bằng pgAdmin/DBeaver, thử insert 1 dòng age=150 để xác nhận CHECK constraint chặn được.

**Kết quả buổi**: 4 bảng rỗng, có khóa chính/khóa ngoại và CHECK constraint đúng.

---

## Buổi 3-4 — ETL nạp dữ liệu thật (historical bulk load) 🔧

**Prompt AI**:
> "Viết script Python dùng pandas + SQLAlchemy đọc file Credit_Risk_Dataset.xlsx, tách cột theo schema 4 bảng [dán schema]. Trước khi insert: (1) set NaN cho person_age > 100 và cho person_emp_length > (person_age - 14) — đây là lỗi dữ liệu thật cần chặn trước khi impute; (2) median imputation cho person_emp_length và loan_int_rate SAU bước chặn outlier; (3) bỏ cột loan_to_income_ratio (trùng loan_percent_income); (4) sinh loan_date rải đều ngẫu nhiên trong 24 tháng gần nhất và application_ref = uuid4 cho mỗi dòng, gắn data_source='historical'. Ghi vào PostgreSQL theo transaction, insert cities trước để lấy city_id."

- Buổi 3: viết script, insert thử với 1000 dòng đầu, kiểm tra dữ liệu vào đúng bảng, xác nhận không còn `person_age > 100`.
- Buổi 4: chạy full 32,581 dòng, viết query kiểm tra đếm số dòng mỗi bảng khớp nhau, xác nhận `loan_date` rải đều qua 24 tháng và `data_source='historical'` toàn bộ. Sau khi xác nhận đúng, chụp snapshot để có thể mang sang máy khác:
  ```bash
  bash scripts/dump_db.sh   # ghi ra db-seed/01_seed.sql
  ```
  Từ giờ, `docker compose down -v && docker compose up -d` trên bất kỳ máy nào có repo này sẽ tự nạp lại đúng 32,581 hồ sơ, không cần chạy lại ETL.

**Kết quả buổi**: toàn bộ 32,581 hồ sơ nằm trong Postgres, phân bổ đúng 4 bảng, không còn outlier tuổi/thâm niên vô lý, có snapshot portable.

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
> "Viết 1 câu SQL PostgreSQL dùng CTE nối 4 bảng cities, customers, credit_bureau, loans; **bắt buộc thêm WHERE data_source = 'historical'** trong CTE gốc để loại hồ sơ synthetic_daily chưa có outcome thật; thêm cột window function tính loan_amnt trung bình theo nhóm tuổi (age bucket 10 năm) và tỷ lệ default trung bình theo loan_grade."

- Buổi 6: viết CTE nối bảng, xác nhận số dòng ra đúng bằng số hồ sơ historical (không lẫn synthetic).
- Buổi 7: thêm window functions (avg theo age bucket, tỷ lệ default theo grade, rank theo credit_utilization_ratio trong từng quốc gia), test kết quả bằng tay trên vài dòng mẫu.

**Kết quả buổi**: 1 câu SQL trả về bảng feature đầy đủ, chỉ chứa dữ liệu lịch sử có nhãn thật, sẵn sàng đọc vào Python bằng SQLAlchemy.

⚠️ **Lưu ý cho buổi 9-11**: 3 cột window function (`avg_loan_amnt_by_age_bucket`, `default_rate_by_grade`, `util_rank_in_country`) là cột tổng hợp cho dashboard (buổi 15-16), KHÔNG phải input cho model. `default_rate_by_grade` tính trực tiếp bằng `AVG(loan_status)` — tức là chính target — trên toàn bộ tập historical trước khi chia train/test, nên dùng làm feature sẽ là leakage nghiêm trọng hơn cả `loan_grade`. Đây là fact về chính câu SQL bạn vừa viết (đúng bởi định nghĩa), không phải insight cần "khám phá" — khác với buổi 8 bên dưới.

---

## Buổi 8 — EDA & xử lý missing 🔧 (viết lại — chỉ dạy phương pháp, không nêu sẵn kết quả)

- Đọc dữ liệu từ câu SQL ở buổi 6-7 vào pandas.
- Vẽ phân phối `loan_status`, phân tích pattern missing của `loan_int_rate` (missing ngẫu nhiên hay có quy luật theo cột khác?).
- **Tự phát hiện leakage bằng phương pháp, không tra số liệu có sẵn ở đâu cả**:
  - Với mỗi cột phân loại (`loan_grade`, `home_ownership`, ...), tự tính default rate theo từng nhóm. Nhóm thấp nhất và nhóm cao nhất càng gần 2 đầu 0%/100% (near-perfect separation) thì càng đáng nghi leakage — quan trọng là nhận ra *pattern* này, con số chính xác của dataset này không cần nhớ trước.
  - Với các cặp cột numeric-categorical, tự tính variance của cột numeric trong từng nhóm của cột categorical. Variance gần 0 nghĩa là cột numeric gần như tất định theo cột kia — 2 cột đang mã hoá cùng 1 thông tin.
  - Với mỗi cột nghi ngờ: cột này có sẵn TẠI THỜI ĐIỂM cần ra quyết định không, hay được hệ thống sinh ra SAU quyết định (chấm điểm/underwriting)? Đây là câu hỏi domain, cần tự suy luận, không phải tra cứu.
- Xác nhận lại không còn `person_age`/`person_emp_length` bất thường (đã chặn ở ETL buổi 3-4, đây là bước kiểm tra chéo, không phải phát hiện mới).
- **Tự ghi lại kết luận** (cột nào bị loại khỏi feature set at-application, vì sao) — buổi 9 sẽ dùng đúng danh sách này, không phải danh sách có sẵn từ kế hoạch.

**Kết quả buổi**: dataset sạch, có quyết định về feature set cho buổi 9-11 — quyết định này là của bạn (hoặc agent tự tính trên dữ liệu thật), không phải copy từ tài liệu.

---

## Buổi 9 — Baseline model + formalize tiền xử lý bằng sklearn Pipeline 🔧

**Prompt AI**:
> "Đọc dữ liệu qua view `ml_features` (SELECT * FROM ml_features, xem sql/views.sql) — view này đã tách riêng khỏi `dashboard_aggregates` nên không có sẵn 3 cột window function (avg_loan_amnt_by_age_bucket, default_rate_by_grade, util_rank_in_country); đây là cột tổng hợp cho dashboard, default_rate_by_grade tính trực tiếp từ loan_status nên là leakage nếu dùng làm feature — không cần tự loại bằng tay nữa, việc tách 2 view đã đảm bảo điều này ở tầng schema. Viết 2 pipeline scikit-learn Logistic Regression với class_weight='balanced', train/test split stratify theo loan_status: (a) *portfolio* — dùng đầy đủ feature; (b) *at-application* — loại bỏ các cột bạn đã xác định là leakage ở buổi 8 (dựa trên phân tích của chính bạn/agent ở buổi trước, không phải danh sách có sẵn). Đánh giá cả 2 bằng ROC-AUC và Precision-Recall AUC, in bảng so sánh chênh lệch.
>
> Tiền xử lý PHẢI đóng gói trong `sklearn.pipeline.Pipeline` + `ColumnTransformer`, không xử lý tay bằng pandas rồi mới đưa vào model: `OneHotEncoder` cho các cột categorical dạng nominal (`home_ownership`, `loan_intent`, `gender`, `marital_status`, `education_level`, `employment_type`, `default_on_file`, `country`), `StandardScaler` cho cột numeric — cả hai đặt trong `ColumnTransformer`, gọi `Pipeline.fit(X_train, y_train)` sau khi đã split để đảm bảo cả 2 bước này chỉ học tham số (mean/std, danh sách category) từ tập train."

- **Tự viết phần `ColumnTransformer`/`Pipeline` này** (không để AI generate toàn bộ) — mục tiêu là giải thích được từng quyết định khi phỏng vấn, không chỉ chạy được code:
  - Vì sao `StandardScaler` phải nằm **trong** Pipeline và fit **sau** khi chia train/test, không phải trước? Nếu scale trên toàn bộ dataset rồi mới split, mean/std đã "nhìn thấy" cả tập test — thông tin phân phối của test rò vào train, đánh giá model sẽ lạc quan giả tạo. `Pipeline.fit(X_train, ...)` tự đảm bảo đúng thứ tự này; đừng gọi `scaler.fit(X)` trên toàn bộ `X` rồi mới `train_test_split`.
  - Vì sao `OneHotEncoder` chứ không phải `LabelEncoder` cho biến categorical **nominal** (không có thứ tự tự nhiên, ví dụ `home_ownership`, `loan_intent`)? `LabelEncoder` gán số nguyên 0,1,2,... — model tuyến tính (Logistic Regression) sẽ hiểu nhầm đây là quan hệ thứ tự/khoảng cách (RENT=0 "gần" OWN=1 hơn MORTGAGE=2), trong khi thực tế các nhóm này không có thứ tự. `OneHotEncoder` tách mỗi category thành 1 cột nhị phân độc lập, không áp đặt thứ tự giả.
  - Pipeline này tái sử dụng nguyên vẹn cho buổi 10-11 (XGBoost) và được lưu trong bundle ở buổi 12 (`preprocessor` là chính `ColumnTransformer` này) — viết đúng ngay từ buổi 9 để không phải viết lại.

**Kết quả buổi**: 2 baseline, thấy rõ bằng số liệu mức độ leakage giữa 2 feature set — dùng để quyết định model nào đưa vào Streamlit ở buổi 13. Có 1 `ColumnTransformer`/`Pipeline` tự viết, giải thích được lý do từng bước tiền xử lý.

---

## Buổi 10-11 — Model chính: XGBoost 🔧

**Prompt AI**:
> "Viết pipeline scikit-learn: train XGBoost cho cả 2 feature set (portfolio và at-application, theo đúng danh sách cột đã chốt ở buổi 9) với scale_pos_weight tính từ tỷ lệ class (không dùng SMOTE trước), đánh giá bằng ROC-AUC và PR-AUC, so sánh với 2 baseline Logistic Regression tương ứng."

- Buổi 10: train XGBoost với `scale_pos_weight` cho cả 2 bản, so sánh với baseline.
- Buổi 11: thử thêm SMOTE (imbalanced-learn) cho bản *at-application* (bản thực sự dùng trong Streamlit), so sánh 2 cách xử lý imbalance, chọn cách tốt hơn dựa trên PR-AUC.

**Kết quả buổi**: model tốt nhất đã chọn cho từng mục đích, có bảng so sánh baseline vs XGBoost vs XGBoost+SMOTE, tách rõ theo feature set.

---

## Buổi 12 — Explainability (SHAP) 🔧

**Prompt AI**:
> "Viết code dùng thư viện shap để vẽ summary plot cho model XGBoost *at-application* đã train, và force plot cho 1 hồ sơ cụ thể để giải thích quyết định duyệt/từ chối."

- Xác định 3-5 feature quan trọng nhất theo SHAP (bản at-application).
- Lưu model dạng **bundle**, không chỉ model trần: `joblib.dump({"model": model, "preprocessor": preprocessor, "feature_names": features, "trained_at": ..., "metrics": {...}}, 'models/at_application_model.pkl')` — lặp lại tương tự cho `models/portfolio_risk_model.pkl`.

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

- Viết README: mô tả bài toán, kiến trúc hệ thống (Excel → Postgres → SQL feature engineering → 2 model → Streamlit), **giải thích rõ vì sao có 2 model (at-application vs portfolio) và leakage của loan_grade/loan_int_rate** — dùng chính lý luận bạn tự rút ra ở buổi 8, không phải copy từ kế hoạch này.
- Chụp screenshot/quay video ngắn demo app, gắn vào README.
- Push code lên Github, tách rõ `/etl`, `/sql`, `/notebooks`, `/app`, `/models`, `/tests`, `/scripts`, `/db-seed`.
- Viết bảng so sánh model (Logistic vs XGBoost vs XGBoost+SMOTE, **× 2 feature set**) kèm lý do chọn PR-AUC làm metric chính và lý do loại loan_grade/loan_int_rate khỏi bản dùng thật.

**Kết quả buổi**: project hoàn chỉnh, sẵn sàng đưa vào CV/portfolio, có luận điểm rõ ràng về leakage — và bạn thực sự tự rút ra được luận điểm đó, không chỉ đọc lại kế hoạch.

---

## Nguyên tắc làm việc với AI agent

Mỗi buổi chỉ đưa **đúng 1 prompt** (đã ghi sẵn ở từng buổi) cho AI agent, chạy thử và kiểm tra kết quả trước khi copy sang buổi tiếp theo. Không gộp nhiều buổi vào 1 prompt — dễ ra code rối, khó debug.

Các buổi đánh dấu 🔧 có prompt dài hơn bản gốc vì đã gộp sẵn yêu cầu sửa lỗi kỹ thuật (cap outlier, backfill ngày, application_ref, lọc data_source, tách feature set) — dán nguyên văn, đừng rút gọn. Đây là fix kỹ thuật (kiến trúc/DDL/ETL), không phải kết quả phân tích, nên không có vấn đề "lộ đáp án".

**Riêng buổi 8-9 thì khác**: đây là 2 buổi luyện phân tích, không phải buổi build kiến trúc. Nếu bạn paste sẵn kết quả EDA (số liệu cụ thể) vào prompt, agent chỉ lặp lại chứ không tự tính — buổi học mất tác dụng, và bạn sẽ không có gì thật để nói nếu bị hỏi "sao bạn phát hiện ra điều này" lúc phỏng vấn. Ở 2 buổi này, chỉ đưa *phương pháp* (cách kiểm tra leakage, cách phát hiện outlier), không đưa *kết quả*.

**Về `CLAUDE.md`/`MEMORY.md`** (nếu dùng Claude Code): mục đích của 2 file này là để agent không phải tính lại facts *đã biết* — đúng và nên dùng, nhưng chỉ sau khi facts đó thực sự đã được biết qua buổi 8-9 của chính bạn. Nếu 2 file này có mặt trong repo từ buổi 1, Claude Code sẽ tự đọc chúng ở đầu mỗi phiên và "biết trước" đáp án buổi 8-9 dù prompt buổi đó không nói gì cả. **Chỉ thêm `CLAUDE.md`/`MEMORY.md` vào repo từ buổi 10 trở đi** — sau khi bạn đã tự chốt được feature set ở buổi 8-9. Trước đó, nếu cần agent hỗ trợ debug, xoá tạm 2 file này khỏi thư mục làm việc.
