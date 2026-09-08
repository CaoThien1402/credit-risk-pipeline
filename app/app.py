"""Buổi 13-16: Streamlit app 2 tab.

SỬA quan trọng: 2 model bundle riêng biệt —
  models/at_application_model.pkl  -> Tab 1, KHÔNG có loan_grade/loan_int_rate làm input
  models/portfolio_risk_model.pkl  -> chỉ dùng nội bộ cho Tab 2 (phân tích danh mục đã có)
Không được dùng chung 1 model cho cả 2 mục đích — xem ghi chú leakage trong sql/schema.sql.
"""
import streamlit as st

st.set_page_config(page_title="Credit Risk & Loan Approval", layout="wide")

tab_predict, tab_dashboard = st.tabs(["Dự đoán hồ sơ mới", "Dashboard quản lý"])

with tab_predict:
    st.header("Dự đoán duyệt/từ chối hồ sơ vay mới")
    st.caption(
        "Model tại đây KHÔNG dùng loan_grade/loan_int_rate làm input — "
        "hai cột đó là kết quả chấm điểm rủi ro, không phải dữ liệu có sẵn "
        "tại thời điểm nộp hồ sơ."
    )
    # TODO (buổi 13): form nhập tuổi, thu nhập, số tiền vay, mục đích vay,
    # home_ownership, emp_length, các cột credit_bureau — KHÔNG thêm loan_grade/loan_int_rate.
    # TODO: load_model("models/at_application_model.pkl")
    # TODO (buổi 14): SHAP force plot cho từng dự đoán, hiển thị top 3 lý do.
    st.info("TODO buổi 13-14: form input + load model + SHAP force plot.")

with tab_dashboard:
    st.header("Dashboard quản lý danh mục")
    # TODO (buổi 15): Plotly đọc trực tiếp Postgres (WHERE data_source='historical'
    # cho số liệu lịch sử; có thể union thêm synthetic_daily nếu muốn theo dõi luồng mới,
    # nhưng phải tách rõ 2 nguồn trong chú thích biểu đồ).
    # TODO (buổi 16): scatter_geo theo cities.latitude/longitude (join qua customers.city_id).
    st.info("TODO buổi 15-16: biểu đồ dư nợ theo tháng, default theo grade/country, scatter_geo.")
