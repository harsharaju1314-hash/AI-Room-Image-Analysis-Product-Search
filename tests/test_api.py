import io
import pytest
from PIL import Image


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert data["model_loaded"] is True
    assert data["database_connected"] is True


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "documentation" in data


def test_analyze_valid_image(client, valid_image_bytes):
    files = {"file": ("living_room.jpg", valid_image_bytes, "image/jpeg")}
    response = client.post("/analyze?top_k=2", files=files)

    assert response.status_code == 201
    data = response.json()

    # Validate response schema
    assert "id" in data
    assert data["filename"] == "living_room.jpg"
    assert data["room_type"] in ["living_room", "bedroom", "kitchen", "bathroom"]
    assert "confidence" in data
    assert "class_probabilities" in data
    assert "image_metrics" in data
    assert data["image_metrics"]["shannon_entropy"] > 0
    assert "similar_products" in data
    assert isinstance(data["similar_products"], list)
    assert len(data["similar_products"]) <= 2


def test_analyze_invalid_extension(client):
    files = {"file": ("malicious_script.sh", b"echo 'hello world'", "text/plain")}
    response = client.post("/analyze", files=files)

    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


def test_analyze_oversized_image(client, oversized_image_bytes):
    files = {"file": ("large_room.jpg", oversized_image_bytes, "image/jpeg")}
    response = client.post("/analyze", files=files)

    assert response.status_code == 413
    assert "exceeds maximum allowed limit" in response.json()["detail"]


def test_analyze_corrupt_image(client):
    corrupt_bytes = b"NOT_AN_IMAGE_PAYLOAD_CORRUPT"
    files = {"file": ("broken.jpg", corrupt_bytes, "image/jpeg")}
    response = client.post("/analyze", files=files)

    assert response.status_code == 400
    assert "corrupt or is not a valid image" in response.json()["detail"]


def test_analyze_missing_file(client):
    response = client.post("/analyze")
    assert response.status_code == 422


def test_get_analysis_by_id(client, valid_image_bytes):
    # First create an analysis
    files = {"file": ("test_room.jpg", valid_image_bytes, "image/jpeg")}
    post_res = client.post("/analyze", files=files)
    analysis_id = post_res.json()["id"]

    # Now retrieve it
    get_res = client.get(f"/analysis/{analysis_id}")
    assert get_res.status_code == 200
    data = get_res.json()
    assert data["id"] == analysis_id
    assert data["filename"] == "test_room.jpg"


def test_get_analysis_not_found(client):
    response = client.get("/analysis/999999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_get_product_by_id(client, db_session):
    from app.models.entities import Product
    p = Product(
        name="Sample Chair",
        category="chair",
        room_type="living_room",
        price=150.0,
        description="Sample wooden chair"
    )
    db_session.add(p)
    db_session.commit()
    db_session.refresh(p)

    response = client.get(f"/products/{p.id}")
    assert response.status_code == 200
    assert response.json()["name"] == "Sample Chair"


def test_get_product_not_found(client):
    response = client.get("/products/999999")
    assert response.status_code == 404
