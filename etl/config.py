"""Postgres connection config, read from environment variables (.env)."""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()


def get_engine():
    """Build a SQLAlchemy engine from DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD env vars."""
    url = (
        f"postgresql+psycopg2://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
        f"@{os.environ.get('DB_HOST', 'localhost')}:{os.environ.get('DB_PORT', 5432)}"
        f"/{os.environ['DB_NAME']}"
    )
    return create_engine(url)
