"""Cấu hình kết nối, đọc từ biến môi trường (.env). Dùng chung cho historical_load.py và daily_ingest.py."""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()


def get_engine():
    """Tạo SQLAlchemy engine kết nối PostgreSQL từ biến môi trường.

    Biến cần có (xem .env.example):
      DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    """
    url = (
        f"postgresql+psycopg2://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
        f"@{os.environ.get('DB_HOST', 'localhost')}:{os.environ.get('DB_PORT', 5432)}"
        f"/{os.environ['DB_NAME']}"
    )
    return create_engine(url)
