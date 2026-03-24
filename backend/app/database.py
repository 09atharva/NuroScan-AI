"""
SQLAlchemy database setup and models for brain tumor classification.
"""
import os
from datetime import datetime
from dotenv import load_dotenv

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, JSON, create_engine
)

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Load .env file
load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/brain_tumor_db"
)

# Neon / Aiven / Heroku provide URLs starting with postgres:// but
# SQLAlchemy requires postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Use SSL for cloud-hosted databases (those with non-localhost hosts)
connect_args = {}
if "localhost" not in DATABASE_URL and "127.0.0.1" not in DATABASE_URL:
    connect_args = {"sslmode": "require"}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ScanRecord(Base):
    """Stores each prediction result."""
    __tablename__ = "scan_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255))
    tumor_type = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    severity = Column(String(20))
    details = Column(JSON)  # Full softmax scores per class
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Create all tables (dev convenience — use Alembic for production)."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency that provides a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
