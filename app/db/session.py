import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings
from app.models.entities import Base

logger = logging.getLogger(__name__)

# Determine if we should attempt PostgreSQL or SQLite for fallback
engine_url = settings.DATABASE_URL
is_sqlite = engine_url.startswith("sqlite")

try:
    if is_sqlite:
        engine = create_engine(engine_url, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(engine_url, pool_pre_ping=True, pool_size=10, max_overflow=20)
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as e:
    logger.warning(f"Could not connect to configured database ({engine_url}): {e}. Creating in-memory fallback engine.")
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """
    Initializes PostgreSQL extensions (pgvector) and creates tables.
    """
    try:
        with engine.connect() as connection:
            if not str(engine.url).startswith("sqlite"):
                try:
                    connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                    connection.commit()
                    logger.info("pgvector extension initialized.")
                except Exception as ext_err:
                    logger.warning(f"Could not initialize pgvector extension: {ext_err}")
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables verified/created successfully.")
    except Exception as err:
        logger.warning(f"Database initialization warning (offline DB): {err}")


def get_db():
    """
    FastAPI dependency for yielding database session with automatic cleanup.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
