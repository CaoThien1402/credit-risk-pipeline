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


---
---

# Review lần 2 — trước khi sang buổi 10

**Ngày**: 2026-09-12 (sau khi đã xử lý #1, #3, #8)
**Bối cảnh**: buổi 9 giờ đã có code thật và có số thật, nên lần review này soi được những thứ lần trước chỉ suy đoán.
**Cách đọc**: 4 pass, mỗi pass ghi rõ đang dùng lăng kính nào. Cùng một file có thể bị soi 2 lần bằng 2 lăng kính khác nhau và ra 2 kết luận khác nhau — đó là chủ ý.

## Đã xử lý từ review lần 1

| # | Nội dung | Trạng thái |
|---|---|---|
| #1 | 7 cột chưa quét leakage | ✅ Đã quét toàn bộ 22 cột, dtype-driven. **Không có leakage.** Chi tiết ở pass 4 |
| #2 | `loan_grade` vào nhánh numeric → crash | ✅ PR #8 |
| #3 | `preprocessing.py` còn là stub | ✅ Đã implement, có số thật. Rule "tự viết tay" đã gỡ khỏi CLAUDE.md theo quyết định của bạn |
| #4 | CI chạy trùng + không có Postgres | ✅ PR #11 |
| #5 | Test coverage | ✅ 2 → 39 test |
| #6 | `requirements.txt` lệch | ✅ PR #13, đã xoá |
| #7 | `daily_ingest` không tái lập được | ✅ PR #12 |
| #8 | Commit cả xlsx lẫn seed | ✅ Giữ cả hai, đã ghi lý do + cái giá vào README |
| #9 | Tài liệu lệch code | ✅ Rải rác PR #11-#14 |
| #10 | CLAUDE.md ra đời trước buổi 8 | ⚠️ Không sửa được (lịch sử), đã ghi nhận |
| #11 | Daily ingest hỏng im lặng | ⚠️ Logging đã sửa (PR #10), **nguyên nhân gốc vẫn chưa rõ** |

---

## Pass 1 — TECH DEBT AUDIT

*(Lăng kính: "cái gì sẽ cắn tôi ở buổi 10-11, cái gì chỉ là xấu mặt")*

Backlog, không phải văn xuôi. Xếp theo mức độ sẽ gây đau thật.

### P0 — sẽ cắn trong buổi 10-11

**P0-1. `models/` chưa có gì, nhưng buổi 12 phụ thuộc vào format bundle.**
`model/bundle.py::validate_bundle` đã tồn tại và có 10 test, nhưng **chưa có code nào gọi nó**. Buổi 12 mới `joblib.dump(...)`. Rủi ro: viết xong bundle rồi mới phát hiện thiếu key, lúc đó `app.py` buổi 13 mới chết. Cách chặn: gọi `validate_bundle(bundle)` ngay trước `joblib.dump` ở buổi 12 — 1 dòng.

**P0-2. Trần của dataset thấp hơn bạn tưởng, và điều đó đổi kỳ vọng buổi 10.**
9/22 feature là nhiễu thống kê (xem pass 4). Nghĩa là XGBoost buổi 10 sẽ **không** nhảy vọt so với 0.8072 — kỳ vọng hợp lý là +0.01~0.03 ROC-AUC. Nếu buổi 10 ra 0.95 thì đó là **dấu hiệu bug/leakage**, không phải thành công. Ghi trước con số kỳ vọng vào đây để buổi 10 không tự lừa mình.

**P0-3. Ngưỡng quyết định duyệt/từ chối vẫn chưa chọn.** Buổi 13 cần nó. `predict_proba` mặc định cắt 0.5 là **sai** với base rate 21.8% cộng `class_weight="balanced"` — model đã được hiệu chỉnh lệch, nên 0.5 không còn là điểm trung tính. Đây là quyết định của bạn (cost của false negative vs false positive), không phải thứ agent nên tự chọn.

### P1 — nên xử trước khi đi xa hơn

**P1-1. Nguyên nhân `daily_ingest` fail 09-10/09-11 vẫn chưa biết.** Logging đã sửa nên lần sau sẽ rõ, nhưng hiện tại pipeline "tự động" của bạn đã **3 ngày không chạy được** (09-10, 09-11, và 09-12 chưa tới 20:00). Không có cảnh báo. Nếu kể chuyện này lúc phỏng vấn, câu hỏi kế tiếp chắc chắn là *"làm sao bạn biết khi nó hỏng?"*.

**P1-2. `etl/historical_load.py:24` còn TODO về chiến lược impute** (median vs KNN). Buổi 8 đã kết luận median là hợp lý (missingness là MCAR), nhưng TODO chưa gỡ → người đọc tưởng còn bỏ ngỏ.

**P1-3. `app/app.py` có 7 TODO, toàn bộ buổi 13-16 chưa bắt đầu.** Không phải nợ — đúng tiến độ. Ghi ở đây chỉ để backlog đầy đủ.

### P2 — cosmetic, không chặn gì

- `notebooks/01_eda.ipynb` cell 6 vẫn đọc `../data/Credit_Risk_Dataset.xlsx` trực tiếp để phân tích missingness gốc. Hợp lý (Postgres đã impute mất rồi), nhưng là chỗ duy nhất còn đọc Excel — nếu đổi đường dẫn file thì nhớ chỗ này.
- `db-seed/01_seed.sql` đang lệch với DB sống (seed chụp 2026-09-10, DB giờ có thêm 79 dòng synthetic). Vô hại vì `ml_features` lọc chúng ra.

### Đã đóng — không còn là nợ

- ~~`sql/views.sql` chưa tách~~ → đã tách, có 5 test cấu trúc canh trong CI.
- ~~Python 3.14 rủi ro pin~~ → **đóng hẳn**. xgboost 3.4.1, shap 0.52.0, sklearn 1.9.0, imblearn 0.14.2 đều bản stable, import sạch, không workaround. Không cần downgrade.
- ~~`tests/` không canh thứ tự fit của ColumnTransformer~~ → có `test_notebook_does_not_fit_before_train_test_split`, đã verify 2 chiều.

---

## Pass 2 — ARCHITECTURE REVIEW

*(Lăng kính: "thiết kế này còn đứng vững không khi code thật đã tồn tại")*

Kiến trúc: Postgres (4 bảng) → SQL feature layer (2 view) → 2 model bundle → Streamlit.

### Câu hỏi chính: danh sách cột thực tế có khớp với cái CLAUDE.md tuyên bố không?

Chạy thật, không đọc code rồi đoán:

```
portfolio: 20 features (13 num + 7 cat)
  leakage cols present   : ['loan_grade', 'loan_int_rate']
  protected cols present : none
  dashboard cols present : none

at_application: 18 features (12 num + 6 cat)
  leakage cols present   : none
  protected cols present : none
  dashboard cols present : none
```

**Khớp hoàn toàn.** Cả 3 ràng buộc kiến trúc đều đúng ở tầng code thật, không chỉ ở tầng tài liệu:

1. `at_application` không có `loan_grade`/`loan_int_rate` — đúng thiết kế 2 model.
2. Không feature set nào chạm `dashboard_aggregates` — việc tách 2 view là safeguard thật.
3. Protected attributes bị loại khỏi **cả hai** — chặt hơn `LEAKAGE_COLS`, đúng chủ ý.

### Bằng chứng cứng cho "fit chỉ trên train"

```
portfolio      : 20 raw -> 43 cột sau one-hot
  scaler fitted on 13 numeric cols, n_samples_seen = 26064
at_application : 18 raw -> 35 cột sau one-hot
  scaler fitted on 12 numeric cols, n_samples_seen = 26064
train rows = 26064
```

`n_samples_seen_` **bằng đúng** số dòng train (26,064 = 80% của 32,581). Nếu scaler từng nhìn thấy test thì con số này phải là 32,581. Đây là bằng chứng ở tầng object, không phải lời hứa trong comment.

### Chỗ implementation đi lệch khỏi thiết kế tài liệu

**Lệch 1 — `loan_grade` được one-hot dù đã khai là ordinal.**
`model/features.py` tách riêng `ORDINAL_CATEGORICAL_COLS = ["loan_grade"]` với lý do "grade có thứ tự A < B < ... < G", nhưng `split_numeric_categorical` gộp nó chung vào `categorical_cols` và `build_preprocessor` one-hot toàn bộ. Tức là **thông tin thứ tự bị vứt đi** — đúng cái thứ mà việc tách list ra để bảo tồn.

Đánh giá: **không phải bug**, one-hot vẫn đúng và an toàn. Nhưng cái tên `ORDINAL_CATEGORICAL_COLS` hiện đang hứa nhiều hơn code làm. Đây là quyết định mở cho buổi 10-11 (đã ghi trong comment) — chỉ cần đừng quên rằng hiện tại nó **chưa** được đối xử như ordinal.

**Lệch 2 — kiến trúc "2 model bundle" chưa được kiểm chứng end-to-end.**
`app/utils.py::load_model_bundle` cộng `model/bundle.py::validate_bundle` định nghĩa hợp đồng, nhưng chưa có file `.pkl` nào tồn tại. Hợp đồng mới chỉ được test bằng dict giả. Rủi ro thật nằm ở buổi 12, không phải bây giờ.

### Kiến trúc có còn sound không?

**Có.** Điểm mạnh nhất là leakage bị chặn ở **tầng schema** (2 view riêng) chứ không phải tầng quy ước — và giờ có test CI canh nó. Thêm `PROTECTED_ATTRIBUTE_COLS` làm tầng chặn thứ hai theo trục khác (pháp lý thay vì thống kê). Không có gì cần đổi trước buổi 10.

---

## Pass 3 — CODE REVIEW

*(Lăng kính: "review cái diff này trước khi tôi merge" — soi dòng, không nói lại kiến trúc)*

Review diff buổi 9 (`model/preprocessing.py` cộng 2 notebook). Tìm: xử lý lỗi, edge case trong train/test split, và thứ **fail im lặng thay vì fail to**.

### Đã tìm thấy và đã sửa

**C-1. [ĐÃ SỬA] Comment nói ngược hoàn toàn với hành vi thật.**
Tôi viết trong `build_preprocessor`:

```python
# Fail loudly if a column reaches this that belongs to neither list, rather than
# silently dropping it - a dropped feature is invisible in the metrics.
remainder="drop",
```

Kiểm bằng cách chạy thật với một cột `forgotten_col` không nằm trong list nào:

```
input columns : ['age', 'income', 'home_ownership', 'forgotten_col']
output names  : ['age', 'income', 'home_ownership_OWN', 'home_ownership_RENT']
>>> 'forgotten_col' silently dropped: True
```

`remainder="drop"` làm **đúng ngược lại** điều comment tuyên bố. Một feature quên phân loại sẽ biến mất không dấu vết, metrics vẫn đẹp. `ColumnTransformer` không có option "raise nếu gặp cột lạ", nên đã sửa comment nói đúng sự thật và trỏ sang `test_every_feature_column_is_classified` — nơi bảo đảm thật sự nằm.

Đáng chú ý: đây là comment do chính tôi vừa viết ở bước trước. Comment sai nguy hiểm hơn không có comment, vì người sau sẽ tin nó mà không kiểm.

**C-2. [ĐÃ SỬA] Cột nằm trong cả 2 list → nhân đôi âm thầm.**
Nếu một cột lọt vào cả `numeric_cols` lẫn `categorical_cols`, `ColumnTransformer` xử lý nó 2 lần, ma trận rộng thêm, fit vẫn thành công, metrics vẫn bình thường. Đã thêm guard raise `ValueError`.

**C-3. [ĐÃ SỬA] List rỗng → train trên ma trận rỗng.** Đã thêm guard.

**C-4. [ĐÃ SỬA] Docstring test và dead code còn nói về stub đã bị xoá.**
`tests/test_preprocessing.py` còn `except NotImplementedError: pytest.skip(...)` và docstring "While build_pipeline() is still an unwritten stub these skip" — cả hai đã chết sau khi implement. Đã gỡ.

### Đã kiểm, không có vấn đề

- **`max_iter`**: lbfgs mặc định 100 vòng **không hội tụ** trên ma trận 43 cột sau one-hot. Đã đặt 2000 và không còn `ConvergenceWarning`. Nếu để mặc định, sklearn chỉ warning rồi vẫn trả model — kết quả tệ hơn nhưng **không** báo lỗi. Đây đúng là loại "fail im lặng" pass này đi tìm, và nó đã được chặn.
- **`train_test_split`**: có `stratify`, có `random_state=42`. Base rate train/test đều 0.2182 — stratify hoạt động đúng.
- **Chấm điểm**: `predict_proba(test)[:, 1]`, `roc_auc_score(test[TARGET], ...)`. Chỉ trên test, không lẫn train. Đã xác nhận lại ở pass 4 bằng 5-fold CV.
- **`handle_unknown="ignore"`**: có chủ ý, có test (`test_unseen_category_does_not_crash_prediction`), có comment giải thích đánh đổi. Đây là quyết định phỏng vấn hay hỏi — lý do đã nằm ngay cạnh code.

### Còn lại, không chặn merge

- `build_pipeline` hardcode `LogisticRegression`. Buổi 10 cần XGBoost dùng lại đúng `build_preprocessor` này — lúc đó nên tách tham số estimator ra, **đừng** copy-paste `ColumnTransformer` sang file mới (sẽ tạo 2 nguồn sự thật).

---

## Pass 4 — DATA VALIDATION

*(Lăng kính: "QA số này trước khi tôi đem đi trình bày")*

Số buổi 9 sinh ra:

| Feature set | ROC-AUC | PR-AUC | n features |
|---|---|---|---|
| `portfolio` | 0.8713 | 0.7206 | 20 |
| `at_application` | 0.8072 | 0.6240 | 18 |
| **chênh lệch** | **0.0640** | **0.0966** | 2 |

### Kiểm 1 — số này có nằm trên sàn không?

```
base rate (full/train/test) : 0.2182 / 0.2182 / 0.2182
dummy (stratified)          : ROC-AUC 0.5050 | PR-AUC 0.2200
```

Sàn PR-AUC của một model vô dụng **chính bằng base rate** (0.2182) — dummy ra 0.2200, đúng như lý thuyết. Model thật ra 0.6240 và 0.7206, tức **cao hơn sàn 2.9-3.3 lần**. Không phải model giả vờ có tín hiệu.

### Kiểm 2 — có phải ăn may một split đẹp không?

5-fold cross-validation trên toàn bộ dữ liệu:

```
portfolio       ROC 0.8681 +/- 0.0083 | PR 0.7106 +/- 0.0192
at_application  ROC 0.8042 +/- 0.0066 | PR 0.6106 +/- 0.0088
```

Holdout (0.8713 / 0.8072) nằm **trong khoảng nửa độ lệch chuẩn** của trung bình CV (0.8681 / 0.8042). Không phải artifact của `random_state=42`. PR-AUC holdout của `at_application` (0.6240) cao hơn CV mean (0.6106) khoảng 1.5 std — hơi lạc quan một chút nhưng vẫn trong biên bình thường.

### Kiểm 3 — con số có "quá đẹp" hoặc "quá phẳng" không?

**Không có dấu hiệu bất thường.**

- Không cái nào chạm 0.95+ (dấu hiệu leakage còn sót).
- Không cái nào quanh 0.5 (dấu hiệu không có tín hiệu).
- Chênh lệch 0.064 ROC-AUC giữa 2 feature set **đúng độ lớn kỳ vọng**: `loan_grade` là predictor rất mạnh nhưng không phải duy nhất, nên bỏ nó ra phải làm giảm — giảm vừa phải, không sụp về 0.5 mà cũng không gần như không đổi. Cả hai thái cực đều sẽ đáng ngờ.

### Kiểm 4 — PR-AUC giảm nhiều hơn ROC-AUC, có hợp lý không?

Có. PR-AUC giảm 0.0966 còn ROC-AUC chỉ giảm 0.0640. PR-AUC khắt khe hơn trên dữ liệu mất cân bằng vì nó **bỏ qua true negative** — mà 78.2% dữ liệu là negative. Bỏ 2 cột mạnh nhất làm tổn thương khả năng bắt đúng nhóm thiểu số nhiều hơn là làm tổn thương thứ tự xếp hạng tổng thể. **Đây chính là lý do buổi 11 nên lấy PR-AUC làm trọng tài** — và giờ bạn có số của chính mình để nói câu đó.

### Phát hiện lớn nhất của pass này: 9/22 feature là nhiễu

Quét single-feature ROC-AUC (0.5 = nhiễu thuần):

| Cột | AUC đơn lẻ |
|---|---|
| `past_delinquencies` | **0.5004** |
| `open_accounts` | **0.4980** |
| `credit_utilization_ratio` | **0.5051** |
| `loan_term_months` | **0.5074** |

Và default-rate spread của các cột phân loại (base rate 21.8%):

| Cột | Spread |
|---|---|
| `gender` | 0.11 pp |
| `country` | 0.13 pp |
| `marital_status` | 0.59 pp |
| `education_level` | 1.01 pp |
| `employment_type` | 1.10 pp |

**`past_delinquencies` ở AUC 0.5004 là điều đáng nói nhất trong cả lần review này.** Trong một hồ sơ tín dụng thật, lịch sử nợ xấu là một trong những predictor **mạnh nhất** của default. Thấy nó ngang với tung đồng xu nghĩa là cột này (cùng `credit_utilization_ratio` và `open_accounts`) được sinh **độc lập với target** — chúng là dữ liệu bịa, không phải dữ liệu bureau thật.

Hệ quả kép:

- **Tốt cho leakage**: sinh độc lập thì không thể leak. Không phải làm lại buổi 9-11.
- **Xấu cho trần model**: 0.8072 đã gần kịch trần mà dataset này cho phép ở at-application. Và **SHAP ở buổi 12 sẽ gán importance cho các cột này** — đó là fitting nhiễu, phải đọc đúng như vậy khi diễn giải.

### Đối chiếu: cột nào có tín hiệu thật

```
loan_percent_income   0.7208
loan_int_rate         0.7081   (leakage, đã loại khỏi at_application)
debt_to_income_ratio  0.6983
income                0.3098   (đảo chiều — thu nhập cao thì default thấp, đúng trực giác)
```

Tín hiệu thật đến gần như toàn bộ từ **12 cột gốc của bộ Kaggle**, không phải từ phần augment thêm. Đây là câu trả lời trung thực nếu ai hỏi "model của bạn học được gì".

---

## VERDICT: GO — nhưng làm 3 việc trước, liệt kê theo file

**Đi tiếp buổi 10 được.** Không còn blocker kiến trúc, không còn leakage chưa kiểm, baseline đã có số thật và số đó đã qua QA. Rào cản lớn nhất của review lần 1 (*"#1 chưa quét, có thể phải làm lại 4 buổi"*) đã **đóng hẳn** — không có leakage trong 7 cột mới.

3 việc nên làm **trước** khi viết dòng code XGBoost đầu tiên:

1. **`etl/run_daily_ingest.ps1`** — chạy tay một lần để xác nhận bản sửa logging hoạt động trên Task Scheduler thật, và backfill 2 ngày đang thiếu. Hiện pipeline "tự động" đã 3 ngày không có dữ liệu. Đây là việc duy nhất trong danh sách này **không** liên quan tới model, nhưng nó là thứ đang thật sự hỏng.

2. **`model/preprocessing.py`** — tách tham số estimator ra khỏi `build_pipeline` trước khi buổi 10 cần nó. Làm bây giờ tốn 5 phút; làm lúc đang viết XGBoost thì cám dỗ copy-paste `ColumnTransformer` sang file mới rất lớn, và lúc đó bạn có 2 nguồn sự thật cho preprocessing — đúng cái lỗi mà việc tách `sql/views.sql` đã tránh được ở tầng SQL.

3. **`notebooks/02_baseline_model.ipynb`** — ghi lại con số kỳ vọng cho buổi 10 **trước khi chạy**: XGBoost nên cho khoảng 0.82-0.84 ROC-AUC ở `at_application`. Nếu ra trên 0.90, dừng lại và đi tìm bug/leakage thay vì ăn mừng. Viết kỳ vọng trước khi thấy kết quả là cách duy nhất để nó có giá trị.

Ngoài 3 việc trên, `docs/Credit_Risk_Pipeline_Plan_v3.md:146-154` (buổi 10) **không cần sửa gì** — chạy đúng như kế hoạch đã viết.

### Một ghi chú thẳng thắn về #10

Buổi 9 giờ là **code do AI sinh**, theo đúng lựa chọn của bạn, và rule "tự viết tay" đã được gỡ khỏi CLAUDE.md thay vì để lại một rule mà project không còn tuân thủ — đó là lựa chọn đúng, rule chết còn tệ hơn không có rule.

Nhưng hệ quả vẫn còn nguyên: nếu phỏng vấn hỏi *"giải thích vì sao `StandardScaler` phải fit sau split"*, câu trả lời trung thực là **"tôi đọc và hiểu phần này"**, không phải "tôi tự rút ra". Phần lý do đã được viết ngay trong docstring của `model/preprocessing.py` chính là để bạn đọc nó một lần cho kỹ.

Phần **vẫn hoàn toàn là của bạn** và đáng kể trong phỏng vấn: quyết định loại `gender`/`marital_status` vì fair lending, cách đọc con số 0.5004 của `past_delinquencies`, và diễn giải khoảng chênh 0.064/0.097 giữa 2 model. Đó mới là phần khó, và nó không nằm trong bất kỳ prompt nào.


---
---

# GHI CHÚ TRỰC TIẾP — Giải thích số liệu buổi 9 (và buổi 10) khi phỏng vấn

**Mục đích của phần này**: không phải để đọc thuộc lòng. Mỗi mục có 3 phần — con số là gì, tại sao nó ra như vậy, và câu hỏi phỏng vấn viên thường hỏi kèm hướng trả lời. Đọc để hiểu, không phải để chép.

## 1. Base rate 21.8% — vì sao nó là điểm quy chiếu cho mọi thứ khác

Base rate = tỷ lệ default thật trong dữ liệu (7,109/32,581 ≈ 21.8%). Đây không phải một con số phụ — nó là **sàn** để đọc mọi metric khác:

- PR-AUC của một model vô dụng (đoán ngẫu nhiên theo tỷ lệ) **bằng đúng base rate**. Đã verify: `DummyClassifier` cho PR-AUC 0.2200, gần khớp base rate 0.2182.
- ROC-AUC của model vô dụng luôn là 0.5, bất kể base rate — đây là điểm khác biệt quan trọng giữa 2 metric.

**Nếu bị hỏi**: *"Tại sao không dùng accuracy?"* → Với base rate 21.8%, một model luôn đoán "không default" đã đạt 78.2% accuracy mà không học được gì. Accuracy là metric vô dụng trên dữ liệu mất cân bằng này.

## 2. ROC-AUC vs PR-AUC — tại sao dùng cả hai, và tại sao PR-AUC quan trọng hơn

- **ROC-AUC**: khả năng xếp hạng đúng một cặp (1 default, 1 không-default) ngẫu nhiên. Ổn định, dễ hiểu, nhưng **không nhạy với imbalance** — nó tính cả True Negative, mà 78.2% dữ liệu là negative nên rất dễ đạt.
- **PR-AUC**: chỉ nhìn vào precision/recall của lớp positive (default). Bỏ qua True Negative hoàn toàn.

**Con số thật đã thấy (buổi 9)**: bỏ `loan_grade`/`loan_int_rate` làm ROC-AUC giảm 0.064 nhưng PR-AUC giảm 0.097 — giảm nhiều hơn. Đây không phải trùng hợp: 2 cột đó đặc biệt mạnh trong việc phân biệt nhóm default (positive) khỏi nhóm còn lại, nên bỏ chúng làm tổn thương đúng cái PR-AUC đo.

**Nếu bị hỏi**: *"Tại sao PR-AUC là metric chính chứ không phải ROC-AUC?"* → Vì bài toán này quan tâm nhất đến việc **tìm đúng người sẽ default** (positive class), không phải xếp hạng tổng thể. PR-AUC phạt nặng hơn khi model bỏ sót hoặc báo sai nhóm thiểu số — đúng cái ngân hàng quan tâm. Đây cũng chính là lý do buổi 11 chọn PR-AUC làm trọng tài giữa `scale_pos_weight` và SMOTE.

## 3. Khoảng chênh portfolio vs at_application — "cái giá của việc không gian lận"

| | ROC-AUC | PR-AUC |
|---|---|---|
| `portfolio` (giữ `loan_grade`/`loan_int_rate`) | 0.8713 | 0.7206 |
| `at_application` (bỏ 2 cột) | 0.8072 | 0.6240 |
| **chênh lệch** | **0.0640** | **0.0966** |

**Nếu bị hỏi**: *"Tại sao không dùng model portfolio luôn cho điểm cao hơn?"* → Vì 2 cột đó **không tồn tại tại thời điểm** khách hàng nộp đơn — chúng là output của bước underwriting, sinh ra *sau* quyết định mà model này tồn tại để đưa ra. Dùng chúng là feedback loop: model học từ kết quả của chính quy trình nó đang cố thay thế. Con số 0.064/0.097 chính là "giá" phải trả để có một model chạy được thật ở application time — và việc đo được cái giá đó, thay vì chỉ nói suông "có leakage", là phần quan trọng nhất của câu trả lời.

## 4. `class_weight="balanced"` — tại sao không resample

Với base rate 21.8%, nếu train bình thường thì mô hình có rất ít áp lực gradient để học nhóm thiểu số (đoán "không default" cho tất cả đã đúng 78%). `class_weight="balanced"` **rescale loss function** — sai lầm trên mẫu positive bị phạt nặng hơn theo đúng tỷ lệ nghịch với tần suất — mà **không thêm/bớt dòng nào** trong dữ liệu train.

**Nếu bị hỏi**: *"Sao không dùng SMOTE ngay từ đầu cho nhanh?"* → SMOTE tạo ra dòng dữ liệu tổng hợp (nội suy giữa các điểm positive có thật), nghĩa là model học từ những điểm **không thực sự tồn tại** trong dữ liệu gốc. `class_weight`/`scale_pos_weight` không có rủi ro này vì không đụng vào dữ liệu, chỉ đụng vào loss. Đây cũng chính là lý do buổi 10 dùng `scale_pos_weight` trước, và buổi 11 mới so sánh với SMOTE bằng số liệu thật thay vì chọn bừa.

## 5. `StandardScaler` fit sau `train_test_split` — câu hỏi kinh điển

**Nếu bị hỏi**: *"Giải thích vì sao StandardScaler phải fit sau khi split, không phải trước?"*

→ Nếu fit `StandardScaler` trên toàn bộ dữ liệu rồi mới `train_test_split`, thì mean/std dùng để chuẩn hoá đã "nhìn thấy" cả tập test — thông tin phân phối của test rò vào tham số của bước tiền xử lý, dù bản thân model chưa hề thấy nhãn test. Kết quả: điểm đánh giá trên test bị lạc quan giả tạo, vì test không còn thực sự "chưa từng thấy" nữa.

**Điểm nhấn quan trọng** (rút ra trực tiếp từ quá trình làm project này, không phải sách vở): lỗi này **không để lại dấu vết trên object đã fit**. Đã tự kiểm chứng: nhét một `StandardScaler` fit sẵn trên train+test vào `ColumnTransformer` rồi gọi `.fit()` lại — `ColumnTransformer.fit()` **clone và fit lại từ đầu**, trạng thái cũ bị xoá sạch. Nghĩa là không thể viết một unit test kiểm tra "object pipeline" để bắt lỗi này; tín hiệu duy nhất đáng tin là **thứ tự lệnh trong source code** (đã viết `test_notebook_does_not_fit_before_train_test_split` để canh đúng chỗ này). Đây là chi tiết ít người biết và rất đáng nói nếu được hỏi sâu.

## 6. `OneHotEncoder` thay vì `LabelEncoder`

`LabelEncoder` gán số nguyên 0, 1, 2... cho mỗi category. Với biến **nominal** (không có thứ tự tự nhiên, ví dụ `home_ownership`: RENT/OWN/MORTGAGE), việc gán số ngầm định một quan hệ thứ tự và khoảng cách không có thật — model tuyến tính sẽ hiểu "RENT gần OWN hơn MORTGAGE" và "khoảng cách RENT→MORTGAGE gấp đôi RENT→OWN", cả hai đều vô nghĩa. `OneHotEncoder` tách mỗi category thành 1 cột nhị phân độc lập, không áp đặt thứ tự.

**Ngoại lệ đáng nói**: `loan_grade` (A đến G) **thực sự có thứ tự**. Hiện tại project vẫn one-hot nó (an toàn, đúng) nhưng chưa tận dụng tính ordinal — đây là quyết định mở, và biết chỉ ra ngoại lệ này cho thấy hiểu bản chất quy tắc chứ không học thuộc lòng.

## 7. `handle_unknown="ignore"` — đánh đổi thật, không phải mặc định

Form nhập liệu ở buổi 13 hoàn toàn có thể gửi lên một category chưa từng thấy lúc train (ví dụ `loan_intent` mới). Nếu để `OneHotEncoder` mặc định, nó sẽ raise lỗi và sập app trên một input hợp lệ của người dùng. `handle_unknown="ignore"` mã hoá category lạ thành toàn số 0 — app không sập, nhưng dự đoán cho dòng đó kém tin cậy hơn.

**Nếu bị hỏi**: *"Đánh đổi này có ổn không?"* → Ổn trong bối cảnh này vì mọi trường categorical trong form đều là dropdown đóng — category lạ chỉ xảy ra khi dữ liệu train đã cũ (có category mới xuất hiện sau này), không phải do người dùng gõ bừa. Nếu form cho nhập tự do thì cách xử lý này sẽ cần xem lại.

## 8. 9/22 cột là nhiễu thống kê — phát hiện quan trọng nhất, không nằm trong plan gốc

Quét single-feature AUC toàn bộ 22 cột: `past_delinquencies` AUC 0.5004, `open_accounts` 0.4980, `credit_utilization_ratio` 0.5051 — gần như tung đồng xu. Trong dữ liệu bureau thật, đây là những predictor **mạnh nhất** của default. Ở mức 0.5 nghĩa là chúng được sinh **độc lập với target**.

**Nếu bị hỏi**: *"Model của bạn học được gì?"* → Tín hiệu thật đến gần như toàn bộ từ 12 cột gốc của bộ dữ liệu (`loan_percent_income` AUC 0.72, `loan_int_rate` 0.71, `debt_to_income_ratio` 0.70), không phải từ phần dữ liệu được thêm vào sau. Đây là câu trả lời trung thực, và nói được nó cho thấy đã tự phân tích thay vì chỉ chạy model rồi báo cáo AUC.

**Hệ quả cho SHAP (buổi 12)**: SHAP sẽ vẫn gán importance cho `past_delinquencies` dù nó là nhiễu — vì SHAP đo "model dùng cột này nhiều hay ít trong cây quyết định", không đo "cột này có thật sự dự đoán đúng không". Phải phân biệt được 2 khái niệm này khi trình bày SHAP.

## 9. Loại `gender`/`marital_status` — quyết định về pháp lý, không phải thống kê

Khác với `loan_grade` (loại vì leakage), `gender`/`marital_status` bị loại vì lý do **hoàn toàn khác**: đây là protected attributes theo luật tín dụng tiêu dùng (ECOA/Regulation B ở Mỹ, tương đương ở Canada/UK — cả 3 nước đều có trong dataset). Dùng chúng làm input là vấn đề tuân thủ, **bất kể** chúng có dự đoán tốt hay không.

**Nếu bị hỏi**: *"Nếu gender dự đoán rất tốt thì có nên dùng không?"* → Không. Đây chính là điểm phân biệt 2 loại "loại bỏ cột": leakage bị loại vì nó không có sẵn tại thời điểm quyết định; protected attribute bị loại vì luật cấm dùng nó **dù có sẵn và dù dự đoán tốt**. May mắn là ở đây cả hai đều là nhiễu (0.11pp và 0.59pp spread) nên quyết định không phải đánh đổi accuracy, nhưng lập luận phải đứng vững ngay cả khi giả sử chúng có tín hiệu mạnh.

---

## Buổi 10 — XGBoost: 3 điều nên nói được

## 10. `scale_pos_weight` — công thức và vì sao tính trên train, không phải toàn bộ dữ liệu

Công thức: `scale_pos_weight = số dòng negative / số dòng positive`, tính trên **tập train**: 20,378 / 5,686 = **3.5839**. Đây là bản tương đương của `class_weight="balanced"` cho gradient boosting — XGBoost nhân hệ số này vào gradient của các dòng positive.

**Nếu bị hỏi**: *"Sao không tính trên toàn bộ dataset cho chính xác hơn?"* → Vì đó chính là kiểu leakage đã tránh ở `StandardScaler` — tính tỷ lệ class trên cả tập test nghĩa là một hyperparameter huấn luyện "biết trước" phân phối chính xác của test. Cùng một nguyên tắc kỷ luật fit-chỉ-trên-train, áp dụng cho một hyperparameter thay vì một transformer.

## 11. Kết quả buổi 10, và vì sao dự đoán trước đó lại sai

| | ROC-AUC | PR-AUC |
|---|---|---|
| LR baseline (`at_application`) | 0.8072 | 0.6240 |
| XGBoost (`at_application`) | **0.8907** | **0.8055** |

Dự đoán viết trước khi chạy là 0.82-0.84 — **sai**, kết quả thật cao hơn hẳn. Đây không phải điểm yếu để giấu đi, mà là **điểm mạnh để kể**: thay vì chấp nhận con số đẹp và đi tiếp, đã điều tra lại — kiểm tra không có leakage lọt vào, so sánh AUC train (0.9178) vs test (0.8907) để loại khả năng overfit nghiêm trọng, và chạy 5-fold CV.

**Nếu bị hỏi**: *"Bạn tin con số 0.8907 đến mức nào?"* → Đây là câu hỏi hay nhất có thể gặp, và câu trả lời sạch nhất: 5-fold CV cho khoảng **0.8606 ± 0.0355** (dao động 0.804 đến 0.911 giữa các fold) — rộng gấp ~5 lần độ lệch chuẩn của baseline Logistic Regression (±0.0066). Nghĩa là: cải thiện so với LR là **thật** (CV mean 0.8606 vẫn cao hơn hẳn CV mean của LR là 0.8042), nhưng **độ chính xác của riêng con số 0.8907** thấp hơn một số liệu holdout đơn lẻ khiến người nghe tưởng. Trả lời được điều này cho thấy hiểu sự khác biệt giữa "model tốt hơn" và "con số cụ thể đáng tin đến đâu" — phân biệt mà nhiều người làm ML bỏ qua.

## 12. Vì sao chưa dùng SMOTE ở buổi 10

Theo đúng kế hoạch: buổi 10 chỉ dùng `scale_pos_weight`, buổi 11 mới thêm SMOTE và so sánh 2 cách bằng PR-AUC. Lý do tách riêng: nếu làm cả hai cùng lúc, không biết phần cải thiện (hay tệ đi) đến từ thay đổi nào. Đây là thực hành thí nghiệm có kiểm soát — đổi một biến tại một thời điểm.

---

**Ghi chú cuối**: phần lớn code buổi 9-10 do AI viết theo yêu cầu trực tiếp, không phải tự tay gõ. Câu trả lời trung thực nếu bị hỏi "bạn tự viết dòng này à?" là **"tôi hiểu và đã kiểm chứng nó, không tự derive từ đầu"** — còn phần thực sự của riêng mình, đứng vững trong mọi câu hỏi trên, là: đọc con số, phát hiện khi dự đoán sai, biết hỏi "có nên tin số này không" thay vì báo cáo nó, và các quyết định nghiệp vụ (fair lending, chọn PR-AUC, loại leakage). Đó mới là thứ một cuộc phỏng vấn thực sự muốn nghe.
