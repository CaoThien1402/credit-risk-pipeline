"""Buổi 12-14: tiện ích dùng chung cho app Streamlit."""
import joblib


def load_model_bundle(path: str) -> dict:
    """Model bundle PHẢI chứa cả model + preprocessor + feature list + metadata,
    không chỉ model trần — nếu không Streamlit sẽ không tái tạo đúng input transform.

    Cấu trúc mong đợi (lưu ở buổi 12 bằng joblib.dump):
        {
            "model": <trained model>,
            "preprocessor": <ColumnTransformer/pipeline dùng lúc train>,
            "feature_names": [...],
            "trained_at": "...",
            "metrics": {"roc_auc": ..., "pr_auc": ...},
        }
    """
    return joblib.load(path)
