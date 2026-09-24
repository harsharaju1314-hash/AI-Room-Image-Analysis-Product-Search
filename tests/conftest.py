import io
import pytest
from PIL import Image, ImageDraw
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.main import app
from app.db.session import get_db
from app.models.entities import Base, Product
from app.services.vision_model import get_vision_service

# Use in-memory SQLite for fast, isolated unit and integration testing
TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session", autouse=True)
def init_test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def db_session():
    """Provides a transactional database session for each test."""
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    # Seed test products
    vision_service = get_vision_service()
    dummy_img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    emb, _, _, _ = vision_service.process_image(dummy_img)

    p1 = Product(
        name="Test Modern Sofa",
        category="sofa",
        room_type="living_room",
        price=799.0,
        description="Comfortable modern couch",
        image_url="/test/sofa.jpg",
        embedding=emb
    )
    p2 = Product(
        name="Test Platform Bed",
        category="bed",
        room_type="bedroom",
        price=650.0,
        description="Queen size platform bed",
        image_url="/test/bed.jpg",
        embedding=emb
    )
    p3 = Product(
        name="Test Kitchen Stool",
        category="seating",
        room_type="kitchen",
        price=120.0,
        description="Wood island stool",
        image_url="/test/stool.jpg",
        embedding=emb
    )
    session.add_all([p1, p2, p3])
    session.commit()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="function")
def client(db_session):
    """FastAPI TestClient with overridden database session."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def valid_image_bytes():
    """Generates valid JPEG image bytes in memory with distinct textures and lines."""
    img = Image.new("RGB", (300, 300), color=(150, 120, 90))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 250, 250], fill=(20, 40, 80), outline=(255, 255, 255), width=3)
    draw.line([(0, 150), (300, 150)], fill=(200, 200, 50), width=2)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def oversized_image_bytes():
    """Generates an image exceeding the 5MB upload limit."""
    return b"\xFF\xD8\xFF" + b"\x00" * (6 * 1024 * 1024)
