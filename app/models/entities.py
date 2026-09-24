from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text
from sqlalchemy.orm import declarative_base

try:
    from pgvector.sqlalchemy import Vector
    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False

Base = declarative_base()


def utc_now():
    return datetime.now(timezone.utc)


class Product(Base):
    """
    Represents a home-design catalog product with metadata and visual vector embedding.
    """
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False, index=True)
    room_type = Column(String(50), nullable=False, index=True)  # living_room, bedroom, kitchen, bathroom
    price = Column(Float, nullable=False)
    description = Column(Text, nullable=True)
    image_url = Column(String(500), nullable=True)

    # 512-dimensional visual embedding
    if HAS_PGVECTOR:
        embedding = Column(Vector(512), nullable=True)
    else:
        embedding = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class RoomAnalysis(Base):
    """
    Represents a processed room analysis record.
    """
    __tablename__ = "room_analyses"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    room_type = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    class_probabilities = Column(JSON, nullable=False)
    
    # Image metrics computed via scikit-image & OpenCV
    image_metrics = Column(JSON, nullable=False)

    # Matched product IDs stored for historical record retrieval
    matched_product_ids = Column(JSON, nullable=False)

    if HAS_PGVECTOR:
        embedding = Column(Vector(512), nullable=True)
    else:
        embedding = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
