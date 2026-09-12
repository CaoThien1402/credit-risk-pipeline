# Code Review — Credit Risk Pipeline (trạng thái: hết buổi 9)

**Ngày review**: 2026-09-12
**Phạm vi**: toàn repo tại commit `bca5693`, đối chiếu với `docs/Credit_Risk_Pipeline_Plan_v3.md` buổi 1-9.
**Nguyên tắc**: chỗ nào tài liệu và code mâu thuẫn thì **code đúng, tài liệu phải sửa**. Mọi nhận định dưới đây đều dẫn file:line hoặc output lệnh thật.

**Về việc kiểm chứng với DB**: bản review đầu viết khi Docker Desktop đang tắt, nên mọi claim chạm DB đều để trạng thái "chưa xác minh". **Ngày 2026-09-12 đã bật Docker và kiểm lại toàn bộ** — kết quả ở mục "Phụ lục: kiểm chứng trên DB sống" ở cuối file. Tóm tắt: mọi claim về SQL đều đúng, và phát sinh thêm một lỗi vận hành nghiêm trọng (#11) chỉ lộ ra khi query được DB.

---

## Làm tốt

Phần này ngắn, không khen lại nhiều lần.

**1. Kỷ luật train/test trong notebook buổi 9 là đúng.** Đây là thứ tôi soi kỹ nhất và nó sạch:

- `notebooks/02_baseline_model.ipynb` cell 7 dòng 1-3: `train_test_split(df, test_size=0.2, random_state=42, stratify=df[TARGET_COL])` — có `random_state`, có `stratify`.
- cell 10 dòng 6: `pipeline.fit(train_df[cols], train_df[TARGET_COL])` — fit **chỉ trên train**.
- cell 10 dòng 8: `pipeline.predict_proba(test_df[cols])` và dòng 10-11: `roc_auc_score(test_df[TARGET_COL], proba)` / `average_precision_score(test_df[TARGET_COL], proba)` — chấm điểm **chỉ trên test**, không phải train, không phải full set.
- Không có `.fit()` / `.fit_transform()` nào xuất hiện trước `train_test_split` trong bất kỳ cell nào (đã kiểm bằng script quét source, xem mục "Cần cải thiện #2" về việc điều này giờ đã có test canh).

**2. Hai feature set khác nhau đúng bằng 2 cột leakage — đã verify bằng cách chạy thật, không đoán từ tên biến.** Chạy `get_feature_columns` trên đúng danh sách cột của view:

```
only in portfolio: ['loan_grade', 'loan_int_rate']
only in at_app   : []
```

`model/features.py:31-34` — logic chỉ lọc `LEAKAGE_COLS` khi `feature_set == "at_application"`, không có đường nào khác làm 2 list lệch nhau.

**3. `sql/views.sql` tồn tại và phần tách view là thật, không phải comment suông.** `sql/views.sql:38-46` định nghĩa `dashboard_aggregates` như một view riêng `SELECT *` từ `ml_features` rồi mới cộng 3 cột window. Training code query `ml_features` (`02_baseline_model.ipynb` cell 3 dòng 1) nên không có đường nào chạm tới `default_rate_by_grade`. Đây **không** còn là nợ kỹ thuật.

**4. Vài claim trong CLAUDE.md kiểm tra ra đúng:**
- `etl/config.py:6` — `load_dotenv()` gọi ở module level, đúng như tài liệu nói.
- `etl/daily_ingest.py:94` — `ON CONFLICT (application_ref) DO UPDATE`, đúng, không phải `loan_id`.

**5. Python 3.14 không gây ra workaround nào — mối lo này không có bằng chứng.** Toàn bộ package import sạch, đều là bản stable chính thức, không có pre-release pin, không build from source, không suppress warning:

```
xgboost 3.4.1 | shap 0.52.0 | sklearn 1.9.0 | numpy 2.5.2 | pandas 3.0.5 | imblearn 0.14.2
```

**Không cần downgrade Python.** Đóng vấn đề này lại.

**6. Postgres `NUMERIC` trả về `float64`, không phải `Decimal`.** Đây là cái bẫy kinh điển của `pd.read_sql` và nó **không** dính ở đây — bằng chứng là output đã lưu trong `notebooks/01_eda.ipynb` cell 2: `loan_amnt float64`, `income float64`, `credit_utilization_ratio float64`. Không cần cast thủ công.

---

## Cần cải thiện

Xếp theo mức độ nghiêm trọng, nặng nhất trước.

### #1 — [NGHIÊM TRỌNG] 7 cột mới chưa từng được quét leakage, nhưng buổi 9 sẽ train lên chúng

Đây là phát hiện quan trọng nhất của lần review này.

Buổi 8 quét leakage bằng một **danh sách hardcode 5 cột** — `notebooks/01_eda.ipynb` cell 11 dòng 1:

```python
categorical_cols = ['loan_intent', 'loan_grade', 'home_ownership', 'default_on_file', 'country']
```

Sau đó (PR #5, 2026-09-10) view `ml_features` được mở rộng từ 18 → 25 cột, thêm `gender`, `marital_status`, `education_level`, `employment_type`, `open_accounts`, `other_debt`, `loan_term_months`.

Hệ quả cụ thể:

- Feature set `portfolio` hiện có **22 cột**, `at_application` có **20 cột** (số liệu từ chạy thật `get_feature_columns`). Buổi 8 chỉ soi **5** trong số đó.
- 4 cột categorical mới (`gender`, `marital_status`, `education_level`, `employment_type`) **chưa bao giờ** được tính default rate theo nhóm.
- Các cột numeric `open_accounts`, `other_debt`, `loan_term_months`, và cả `credit_utilization_ratio`, `past_delinquencies`, `cred_hist_length` cũng chưa qua bước variance-ratio nào.
- Vì danh sách là hardcode chứ không suy ra từ dtype, nó **âm thầm** bỏ sót mọi cột thêm vào sau. Không có cảnh báo nào.

Thêm một dấu hiệu đáng ngờ cần bạn tự kiểm: bộ credit-risk gốc (lineage Kaggle) chỉ có 12 cột — `person_age`, `person_income`, `person_home_ownership`, `person_emp_length`, `loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate`, `loan_status`, `loan_percent_income`, `cb_person_default_on_file`, `cb_person_cred_hist_length`. Tất cả những cột còn lại trong file 29 cột này (`gender`, `marital_status`, `education_level`, `employment_type`, `open_accounts`, `other_debt`, `credit_utilization_ratio`, `past_delinquencies`, toạ độ thành phố…) là **augment thêm**. Hai khả năng, và chúng dẫn tới kết luận trái ngược nhau:

- Nếu chúng được sinh **ngẫu nhiên, độc lập với target** → chúng là nhiễu thuần. Đưa vào model không leak, nhưng làm loãng SHAP và khiến câu chuyện phỏng vấn của bạn yếu đi ("tại sao model dùng `gender`?" là một câu hỏi khó trả lời cả về kỹ thuật lẫn đạo đức).
- Nếu chúng được sinh **có tham chiếu tới `loan_status`** → đó là leakage nặng, và mọi con số buổi 9-11 sẽ vô nghĩa.

**Tôi cố ý KHÔNG tự chạy phép quét này cho bạn.** Đây đúng là bài tập buổi 8, và nếu tôi chạy rồi đưa số, tôi lặp lại chính xác cái lỗi quy trình mà tôi phản ánh ở #11 bên dưới. Việc cần làm (bạn tự chạy):

```python
# lặp trên dtype, không hardcode danh sách nữa
cat_cols = [c for c in feature_cols if df[c].dtype == 'object' or df[c].dtype.name == 'str']
for col in cat_cols:
    rates = df.groupby(col)['loan_status'].mean() * 100
    print(col, round(rates.max() - rates.min(), 2))   # spread lớn bất thường = nghi ngờ
```

Đồng thời **sửa cell 11 thành suy ra từ dtype** để lần sau thêm cột không bị bỏ sót nữa.

**Hệ quả kèm theo**: output đã commit của `01_eda.ipynb` cell 2 vẫn ghi `Rows, columns: (32581, 18)` — đã lệch với view 25 cột hiện tại. Notebook buổi 8 đang mô tả một dataset không còn tồn tại.

### #2 — [NGHIÊM TRỌNG] `loan_grade` bị phân loại là numeric → pipeline `portfolio` chắc chắn crash (đã sửa ở PR #8)

`loan_grade` là `CHAR(1)` với giá trị `'A'`..`'G'` (`sql/schema.sql:48`), dtype pandas là `str` (bằng chứng: output `01_eda.ipynb` cell 2). Nó nằm trong `LEAKAGE_COLS` (`model/features.py:13`) nhưng **không** nằm trong `NOMINAL_CATEGORICAL_COLS` (`model/features.py:15-24`), trong khi `split_numeric_categorical` (`model/features.py:39-41` bản cũ) phân loại **mọi thứ không có trong list nominal là numeric**.

Chạy thật trên danh sách cột của view, feature set `portfolio`:

```
numeric: ['loan_grade', 'loan_amnt', 'loan_int_rate', ...]   ← loan_grade nằm trong numeric
```

Và tái hiện crash trực tiếp:

```
ValueError: could not convert string to float: 'A'
```

Điểm nguy hiểm: lỗi này **tiềm ẩn**. Nó chỉ nổ vào đúng lúc bạn implement `build_pipeline()` — tức là giữa bài tập buổi 9 — và sẽ trông y như lỗi của chính bạn, khiến bạn mất thời gian debug code mình vừa viết trong khi nguyên nhân nằm ở file khác.

Đã sửa ở PR #8: thêm `ORDINAL_CATEGORICAL_COLS = ["loan_grade"]`, tách riêng khỏi list nominal vì grade **thực sự có thứ tự** (A < B < … < G). One-hot là mặc định an toàn; chuyển sang `OrdinalEncoder` là quyết định mô hình hoá dành cho bạn ở buổi 10-11.

### #3 — [NGHIÊM TRỌNG] CLAUDE.md mô tả code chưa tồn tại như thể đã chạy

`CLAUDE.md:66` viết ở thì khẳng định: *"`OneHotEncoder` for nominal categoricals … `StandardScaler` for numeric columns, inside the same `ColumnTransformer`, fit via `Pipeline.fit(X_train, ...)`"*.

Thực tế `model/preprocessing.py:26` và `:32`:

```python
raise NotImplementedError("Write this yourself - see the module docstring.")
```

**Chưa có `ColumnTransformer` nào tồn tại trong repo này.** Không có `OneHotEncoder`, không có `StandardScaler`, không có con số ROC-AUC/PR-AUC nào.

Đây không phải lỗi nhỏ về câu chữ: người đọc CLAUDE.md (kể cả agent ở phiên sau) sẽ tin rằng buổi 9 đã xong. Theo nguyên tắc "code thắng", CLAUDE.md phải nói rõ đây là **yêu cầu chưa được thực hiện**, không phải mô tả hiện trạng. Câu cuối của đoạn đó có nhắc stub, nhưng phần đầu vẫn đọc như đã implement.

**Nói thẳng: buổi 9 chưa hoàn thành.** Repo mới có scaffold của buổi 9.

### #11 — [NGHIÊM TRỌNG] Pipeline "tự động hằng ngày" đã hỏng 2/3 ngày gần nhất, và log tự xoá bằng chứng

*(Phát hiện bổ sung ngày 2026-09-12 sau khi bật Docker — mức nghiêm trọng ngang #1-#3, chỉ đánh số sau vì tìm ra muộn hơn.)*

Buổi 5 kết luận *"pipeline daily ingestion tự động, idempotent thật"*. Query DB sống:

```
 loan_date  | rows_ingested
------------+---------------
 2026-09-09 |            79
```

**Đúng một ngày duy nhất có dữ liệu**, trong khi hôm nay đã là 2026-09-12.

`logs/daily_ingest.log` cho thấy Task Scheduler **có bắn đúng giờ** 20:00 các ngày 09-10 và 09-11, nhưng **cả hai lần đều fail**, và log chỉ ghi được đúng một dòng vô dụng:

```
2026-09-10T20:00:11 - FAILED: python.exe : Traceback (most recent call last):
    + FullyQualifiedErrorId : NativeCommandError
```

Nguyên nhân nằm ở `etl/run_daily_ingest.ps1:13` (bản cũ): `& $python $script 2>&1 | Out-String` chạy dưới `$ErrorActionPreference = "Stop"` (dòng 3). Trong PowerShell 5.1, mỗi dòng stderr của một native exe bị bọc thành `ErrorRecord`, nên **dòng đầu tiên** của traceback Python đã throw ngay trước khi pipeline kịp chạy xong — `$output` không bao giờ được gán, và catch block chỉ log đúng `ErrorRecord` đầu tiên đó. Python ghi **mọi** traceback ra stderr, nên lỗi này cắt mất nguyên nhân của **tất cả** các lần fail.

Điểm đáng chú ý: `docs/MEMORY.md` ghi bug này **đã được sửa**. Log chứng minh là chưa — lần sửa trước đổi catch block, nhưng throw xảy ra *trước* khi catch nhìn thấy output đầy đủ. Đây là ví dụ rõ nhất trong cả repo cho nguyên tắc "code thắng tài liệu": tài liệu nói đã sửa, log nói chưa, và log đúng.

Đã sửa ở PR #10 (`Start-Process` + 2 file redirect riêng). Verify bằng cách chạy **cả hai** pattern trên cùng một script lỗi:

| Pattern | Log thu được |
|---|---|
| Cũ | `python.exe : Traceback (most recent call last):` + `NativeCommandError` — mất sạch exception |
| Mới | Traceback đầy đủ, kết thúc bằng `ConnectionError: could not connect to server: Connection refused (is Docker running?)`, exit code 1 |

Bản sửa cũng chữa luôn lỗi mojibake trong log (dòng cũ đọc ra `─É├ú upsert 75 hß╗ô s╞í`).

**Vẫn còn bỏ ngỏ**: vì sao 09-10 và 09-11 fail thì **không còn cách nào biết** — bằng chứng đã bị chính bug logging xoá mất. Khả năng cao nhất là Docker Desktop tắt lúc 20:00 (đúng failure mode mà `CLAUDE.md` đã ghi, và hôm nay Docker cũng đang tắt lúc bắt đầu review). Lần fail tới sẽ nói rõ.

**Bài học rộng hơn cho project**: một pipeline "tự động" không có cảnh báo thì hỏng trong im lặng. Bạn chỉ phát hiện ra vì tình cờ query bảng. Nếu định kể chuyện này trong phỏng vấn ("tôi dựng pipeline ingest tự động hằng ngày"), thì câu hỏi tiếp theo gần như chắc chắn là *"làm sao bạn biết khi nó hỏng?"* — và hiện tại câu trả lời là "không biết".

### #4 — [TRUNG BÌNH] CI chạy trùng 2 lần mỗi PR, và không có Postgres nên không test được gì chạm DB

`.github/workflows/ci.yml:3-6`:

```yaml
on:
  push:                      # ← không lọc branch
  pull_request:
    branches: [main]
```

`push` không giới hạn branch, nên mỗi lần push lên feature branch đang mở PR thì **cả 2 trigger cùng bắn**. Bằng chứng quan sát được: `gh pr checks` ở PR #5, #6, #7 đều trả về **2 dòng `test`** với 2 run ID khác nhau. Lãng phí một nửa số phút CI. Sửa: `push: branches: [main]`.

Vấn đề nặng hơn: workflow **không có `services: postgres`**. Nghĩa là CI không thể chạy bất kỳ test nào chạm DB — không test được `ml_features` có đúng 25 cột không, không test được filter `data_source='historical'` còn nguyên không, không test được UPSERT idempotent thật không. Toàn bộ tầng SQL — thứ có nhiều ràng buộc tinh tế nhất trong repo này — hiện **không có một dòng test nào**.

### #5 — [TRUNG BÌNH] Test coverage: trước PR #8 chỉ có 2 test, đều không chạm phần rủi ro nhất

Output thật trước khi tôi thêm test:

```
tests/test_placeholder.py::test_clean_outliers_caps_impossible_age PASSED
tests/test_placeholder.py::test_drop_redundant_columns PASSED
2 passed in 0.80s
```

Cả 2 đều test hàm pandas thuần trong `etl/historical_load.py`. **Không có** test nào cho: `model/features.py`, `model/preprocessing.py`, `etl/daily_ingest.py` (logic UPSERT), `etl/config.py`, các view SQL, hay format model bundle.

Trả lời trực tiếp câu hỏi trong đề bài — *"có test nào fail nếu ai đó chuyển `.fit()` lên trước `train_test_split` không?"*: **Không có.** Và khi viết nó, tôi phát hiện một điều đáng chú ý mà tôi nghĩ bạn nên biết:

> **Test kiểu "kiểm tra `scaler.mean_`" KHÔNG bắt được lỗi này.** Tôi đã thử: dựng một `StandardScaler` fit sẵn trên `train + test`, nhét vào `ColumnTransformer`, rồi chạy test → **test vẫn PASS**. Lý do: `ColumnTransformer.fit()` **clone và fit lại** transformer của nó, nên trạng thái fit sẵn bị vứt đi hoàn toàn, không để lại dấu vết nào trên object đã fit.

Nghĩa là lỗi "scale trước khi split" **không thể bắt ở tầng pipeline** — vì tổn thất đã nằm sẵn trong *dữ liệu* đưa vào `fit()`, không nằm trong *object* pipeline. Tín hiệu duy nhất đáng tin là **thứ tự lệnh trong source notebook**.

PR #8 vì vậy viết `test_notebook_does_not_fit_before_train_test_split`: flatten các code cell của notebook, fail nếu `.fit(`/`.fit_transform(` xuất hiện trước `train_test_split(` đầu tiên. Đã verify hai chiều — pass trên notebook hiện tại, fail đúng khi tôi chèn một cell vi phạm vào bản copy.

Các test khác thêm ở PR #8 (`20 passed, 3 skipped`): regression cho #2, contract cho bundle, và 3 test pipeline-level tự `skip` khi `build_pipeline` còn là stub rồi **tự kích hoạt** khi bạn implement xong — CI xanh ở cả hai trạng thái.

### #6 — [TRUNG BÌNH] `requirements.txt` là bản trùng lặp không pin, lệch với nguồn sự thật

`requirements.txt` liệt kê 12 package **không pin version nào**, và thiếu toàn bộ dev deps (`pytest`, `jupyter`, `kaleido`). Trong khi đó CI (`ci.yml:23`) dùng `uv sync --locked`, tức nguồn sự thật thật sự là `pyproject.toml` + `uv.lock`.

Ai clone repo rồi `pip install -r requirements.txt` sẽ nhận môi trường khác hẳn môi trường CI và khác máy bạn, lại không có công cụ để chạy test. Hoặc xoá file, hoặc sinh nó từ lock file và ghi rõ nó là artifact phái sinh.

### #7 — [TRUNG BÌNH] `daily_ingest.py` không tái lập được, và dùng lẫn 2 kiểu RNG

- `etl/daily_ingest.py:47` — `sample_new_applications(n, reference_stats, seed: int | None = None)` **có** tham số seed.
- `etl/daily_ingest.py:104` — `__main__` gọi `sample_new_applications(n, reference_stats)` — **không bao giờ truyền seed**. Tham số này chết.
- `etl/daily_ingest.py:49` dùng API mới `np.random.default_rng(seed)`, nhưng `:103` lại dùng global legacy `np.random.randint(50, 101)`. Hai hệ RNG khác nhau trong cùng một file; cái ở dòng 103 không chịu ảnh hưởng của seed nào cả.

Không phải bug chặn việc chạy, nhưng nghĩa là không thể tái lập chính xác một ngày ingest nào để debug.

### #8 — [THẤP] Commit cả `.xlsx` lẫn `01_seed.sql`, không chỗ nào giải thích vì sao

Số đo thật: `data/Credit_Risk_Dataset.xlsx` = **6.1M**, `db-seed/01_seed.sql` = **8.8M**, `.git` = **17M**. Tức ~15M/17M lịch sử git là dữ liệu trùng nhau — seed chính là xlsx sau khi qua ETL.

Tôi có grep tìm lời giải thích: `docs/MEMORY.md:30` nói việc commit xlsx là "an explicit user choice", `MEMORY.md:97-99` giải thích mục đích của seed. Nhưng **không chỗ nào** nói vì sao giữ **cả hai**, và cái giá phải trả.

Cái giá đó sẽ tăng dần: mỗi lần chạy `scripts/dump_db.sh` là ghi đè một file 8.8M, và git lưu **vĩnh viễn** blob mới. Vài lần refresh nữa là repo phình lên hàng chục MB cho một project portfolio.

Câu hỏi cần bạn quyết (tôi không tự quyết): giữ cả hai là **có chủ ý** (xlsx = nguồn thô để chứng minh ETL chạy thật; seed = tiện cho người clone) hay chỉ là tiện tay? Nếu có chủ ý thì viết 1 dòng vào README; nếu không, bỏ seed khỏi git và để `docker compose up` chạy ETL.

### #9 — [THẤP] Vài chỗ tài liệu lệch code

- `CLAUDE.md:26` — `tests/ pytest, mirrors etl/ functions`. Sau PR #8 thì `tests/` còn phủ `model/` nữa.
- `CLAUDE.md:20-23` — mô tả `model/` nhưng chưa có `bundle.py` (thêm ở PR #8).
- `README.md:4` — "32,581 rows, 29 columns" mô tả file Excel gốc; view hiện dùng để train có 25 cột. Không sai, nhưng dễ gây nhầm nếu đọc nhanh.
- `model/preprocessing.py:19` — import `LogisticRegression` nhưng chưa dùng ở đâu (vì thân hàm còn là stub). Sẽ hết khi bạn implement.

### #10 — [GHI NHẬN QUY TRÌNH] CLAUDE.md ra đời **trước** buổi 8, nên buổi 8 không phải "tự khám phá" thật

Đây là câu hỏi bạn nhờ trả lời ở mục 1 của đề bài. Kết quả git archaeology:

| Sự kiện | Commit | Thời điểm |
|---|---|---|
| CLAUDE.md + MEMORY.md vào repo | `f20572f` | **2026-09-08 21:12:18** |
| Buổi 8 EDA (`01_eda.ipynb`) | `3940d40` | **2026-09-08 21:26:46** |
| Buổi 9 (`model/`, `02_baseline_model.ipynb`) | `5106c5c` | 2026-09-10 23:26:55 |

CLAUDE.md vào repo **trước buổi 8 đúng 14 phút**. Và nội dung CLAUDE.md **tại chính commit `f20572f` đó** đã chứa sẵn đáp án:

- dòng 35: *"default rate runs ~10% at grade A to ~98% at grade G, and `loan_int_rate` is near-deterministic given grade"*
- dòng 41: *"5 rows with `person_age` 144/123 and 2 rows with `person_emp_length` 123 at age 21-22"*
- dòng 43: *"correlation with `loan_percent_income` is 0.9989"*

Đây đúng là 3 thứ mà buổi 8 lẽ ra phải **tự tìm ra**. Claude Code tự đọc CLAUDE.md ở đầu mỗi phiên, nên tác nhân làm buổi 8 đã có sẵn đáp án trong context trước khi "phân tích".

`docs/Credit_Risk_Pipeline_Plan_v3.md:213` đã cảnh báo đúng tình huống này: *"**Chỉ thêm `CLAUDE.md`/`MEMORY.md` vào repo từ buổi 10 trở đi** — sau khi bạn đã tự chốt được feature set ở buổi 8-9."* Quy tắc này **đã bị vi phạm**.

Mức nghiêm trọng thực tế: không làm hỏng code, và kết luận leakage về `loan_grade` vẫn **đúng về mặt kỹ thuật** (grade đúng là sinh sau underwriting — đây là lập luận domain, không phải thứ tra ra từ số). Nhưng nó làm hỏng **giá trị luyện tập** và, quan trọng hơn cho mục tiêu phỏng vấn: nếu ai hỏi *"bạn phát hiện leakage bằng cách nào?"*, câu trả lời trung thực hiện tại là "nó có sẵn trong file ghi chú", không phải "tôi quét separation rồi thấy grade G ở 98%".

Đây là lý do tôi **không** tự chạy phép quét ở #1 — làm vậy là lặp lại y hệt lỗi này ở quy mô lớn hơn, đúng vào lúc còn kịp sửa.

---

## Next steps

### A. Những thứ cần làm mà kế hoạch 17 buổi **không** có

Dựa trên trạng thái thật của repo, không phải giả định:

1. **Quét leakage cho 7 cột mới** (#1). Kế hoạch giả định buổi 8 đã phủ hết feature set — thực tế nó hardcode 5 cột và view đã đổi sau đó. Không có buổi nào trong 17 buổi quay lại việc này.
2. **Đổi cell 11 của `01_eda.ipynb` sang suy ra từ dtype** thay vì hardcode, để lần sau thêm cột không bị bỏ sót âm thầm.
3. **Thêm `services: postgres` vào CI** + sửa `on: push` thành `branches: [main]` (#4). Kế hoạch không nhắc gì tới CI.
4. **Contract test cho model bundle** (đã làm ở PR #8) — kế hoạch chỉ có đúng một dòng `joblib.dump(...)` ở buổi 12, không có gì bảo vệ format mà `app.py` phụ thuộc.
5. **Quyết định về `requirements.txt`** (#6) và **về việc commit cả xlsx lẫn seed** (#8).
6. **Chọn ngưỡng duyệt/từ chối** cho buổi 13. Kế hoạch nói "hiển thị Duyệt/Từ chối" nhưng không nói ngưỡng nào. Mặc định `0.5` là sai với dữ liệu mất cân bằng 21.8% và là một quyết định thật cần lý do.

### B. Phần nào của buổi 9-11 bạn nên **tự viết**, không để agent sinh

Tiêu chí: đó có phải thứ người phỏng vấn sẽ hỏi *"walk me through why you did X"* và câu trả lời phải là của bạn, không phải nhớ lại từ prompt.

1. **`model/preprocessing.py::build_preprocessor` và `build_pipeline`** (đang là stub). Không phải vì code khó — nó ~10 dòng — mà vì **tham số `handle_unknown` của `OneHotEncoder`** là một quyết định thật: form nhập liệu ở buổi 13 hoàn toàn có thể gửi lên một category chưa từng thấy lúc train. Chọn `'ignore'` hay để nó raise là đánh đổi giữa "app không sập" và "âm thầm dự đoán trên input rác". Người phỏng vấn hỏi cái này rất thường.
2. **Cách encode `loan_grade` ở buổi 10** (nominal one-hot hay `OrdinalEncoder`). PR #8 để ngỏ có chủ ý. Grade **có** thứ tự tự nhiên — đây chính là ngoại lệ của quy tắc "nominal thì one-hot" mà bạn đã học ở buổi 9. Giải thích được *vì sao cột này khác* là thứ phân biệt người hiểu với người thuộc lòng quy tắc.
3. **Công thức `scale_pos_weight` ở buổi 10** và lý do **không** dùng SMOTE trước. Câu hỏi kinh điển: *"tại sao không oversample luôn cho nhanh?"*
4. **Quyết định ở buổi 11: SMOTE hay `scale_pos_weight`, và vì sao lấy PR-AUC làm trọng tài chứ không phải ROC-AUC.** Đây là câu hỏi phỏng vấn phổ biến nhất về dữ liệu mất cân bằng. Câu trả lời phải gắn với con số **bạn tự chạy được**, không phải câu chữ trong kế hoạch.
5. **Diễn giải khoảng chênh giữa `portfolio` và `at_application`** sau khi có số. Con số là output của máy; *"chênh lệch này nghĩa là gì về mặt nghiệp vụ"* là phần của bạn, và là điểm mạnh nhất của cả project khi kể lại.

Ngược lại, những phần **cứ để agent làm**: plumbing SQL/view, wiring ETL, cấu hình CI, test scaffolding, sửa tài liệu. Chúng không phải thứ ai hỏi trong phỏng vấn.

### C. Có nên sang buổi 10 ngay không?

**Không. Cần xử lý #1 trước, và hoàn thành buổi 9 trước.** Lý do cụ thể, không phải thận trọng chung chung:

1. **Buổi 9 thật ra chưa xong.** `model/preprocessing.py:26,32` vẫn là `NotImplementedError`; chưa tồn tại một con số ROC-AUC/PR-AUC nào. Mà buổi 10 (`plan v3:149`) yêu cầu *"so sánh với 2 baseline Logistic Regression tương ứng"* — không có baseline thì không có gì để so.

2. **#1 phải xử trước buổi 10, không phải sau.** Buổi 10-11 train XGBoost trên **đúng feature set đó**. Nếu trong 7 cột chưa quét có một cột leak, thì toàn bộ số liệu buổi 9, 10, 11 đều phải làm lại — và bảng so sánh cuối cùng ở buổi 17 (Logistic vs XGBoost vs XGBoost+SMOTE × 2 feature set) sẽ vô giá trị. Chi phí kiểm bây giờ: khoảng 15 phút. Chi phí phát hiện ở buổi 17: làm lại 4 buổi.

3. **Merge PR #8 trước** để cái crash `loan_grade` không nổ giữa lúc bạn đang implement stub.

**Thứ tự đề xuất:**

1. Merge PR #8 (fix + test).
2. Chạy phép quét leakage dtype-driven cho toàn bộ 22 cột, tự kết luận, cập nhật cell 11 của `01_eda.ipynb`, chạy lại notebook (output hiện tại đang stale ở 18 cột).
3. Tự viết `build_preprocessor`/`build_pipeline`, chạy `02_baseline_model.ipynb` → lúc này 3 test đang skip sẽ tự kích hoạt.
4. Sửa CLAUDE.md cho khớp code (#3, #9), ghi #10 vào MEMORY.md như một process note.
5. Xong hết mới sang buổi 10.

Buổi 10 sau đó chạy **đúng như kế hoạch** — không có gì trong `plan v3:146-154` cần sửa, miễn là feature set đã được kiểm và baseline đã có thật.

---

## Phụ lục: kiểm chứng trên DB sống (2026-09-12)

Bản review đầu viết khi Docker tắt. Sau khi bật, đã query lại toàn bộ. Kết quả:

### Những claim SQL đều đúng

| Claim | Nguồn | Kết quả kiểm chứng |
|---|---|---|
| `ml_features` 25 cột, `dashboard_aggregates` 28 cột | PR #5 | ✅ Đúng chính xác |
| `ml_features` = 32,581 dòng, 0 dòng `loan_status` NULL | `sql/views.sql:34` | ✅ Đúng |
| Filter `data_source='historical'` loại synthetic | `CLAUDE.md` | ✅ **Và filter đang làm việc thật**: bảng `loans` có 32,581 historical + **79 synthetic_daily**, view trả về đúng 32,581 |
| 7 cột thêm ở PR #5 có dữ liệu thật | #1 | ✅ 0 null cả 7 cột; `gender`/`marital_status`/`education_level`/`employment_type` là `str`, `open_accounts`/`loan_term_months` là `int64`, `other_debt` là `float64` |
| `loan_grade` là chuỗi → crash `StandardScaler` | #2 | ✅ dtype `str`, giá trị `['D','B','C','A','E']` — xác nhận trên view 25 cột hiện tại, không chỉ suy từ notebook cũ |
| Bản vá PR #8 có hiệu lực | #2 | ✅ Feature set `portfolio` = 13 numeric + 9 categorical, **không cột non-numeric nào** lọt vào `StandardScaler` |
| `application_ref` là khoá UPSERT thật | `CLAUDE.md` | ✅ Tồn tại constraint `loans_application_ref_key UNIQUE (application_ref)`; `loan_id` chỉ là PK |
| Dòng synthetic có `loan_status` NULL | `CLAUDE.md` | ✅ 79/79 dòng NULL, 79 `application_ref` phân biệt |
| `db-seed/01_seed.sql` phục hồi đúng view 25 cột | PR #5 | ✅ Dòng 165-197 của seed chứa đủ 25 cột (kiểm tĩnh, **không** chạy `down -v` vì sẽ xoá mất 79 dòng synthetic đang có) |

### Claim leakage của `dashboard_aggregates` — đúng đến 6 chữ số thập phân

`CLAUDE.md` nói `default_rate_by_grade` *"is literally `AVG(loan_status)` — the target itself"*. Kiểm bằng cách join 2 view rồi so:

```
 loan_grade | dashboard_col | true_default_rate
------------+---------------+-------------------
 A          |      0.099564 |          0.099564
 D          |      0.590458 |          0.590458
 G          |      0.984375 |          0.984375
```

Khớp tuyệt đối. Dùng cột này làm feature nghĩa là đưa thẳng target vào input. Và kiểm tra cấu trúc: query 3 cột dashboard trong `information_schema` với `table_name='ml_features'` trả về **0 dòng** — tức đường training **không có cách nào** chạm tới chúng. Safeguard là thật, không phải comment suông.

### Phát sinh mới

Xem **#11** ở trên — chỉ lộ ra khi query được `loan_date` của các dòng synthetic.

### Vẫn chưa kiểm

- **Phép quét leakage cho 7 cột mới (#1)**: vẫn để bạn tự chạy, có Docker rồi cũng không đổi — đây là bài tập buổi 8, không phải việc thiếu công cụ.
- **Chạy thật `daily_ingest.py` để backfill**: chưa chạy. Job đã lên lịch sẽ tự bắn lúc 20:00 hôm nay; nếu chạy tay bây giờ thì ngày 2026-09-12 sẽ có 2 batch (mỗi lần sample `client_id` ngẫu nhiên khác nhau → `application_ref` khác nhau → không đè lên nhau). Vô hại vì dòng synthetic bị loại khỏi `ml_features`, nhưng là quyết định của bạn.
- **`docker compose down -v` để test seed end-to-end**: cố ý không chạy, vì sẽ xoá 79 dòng synthetic — lịch sử pipeline thật đang tích luỹ. Đã kiểm tĩnh nội dung seed thay thế.
