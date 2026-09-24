import pytest
import numpy as np
from PIL import Image
from app.services.vision_model import get_vision_service, ROOM_CLASSES


def test_vision_model_singleton():
    service1 = get_vision_service()
    service2 = get_vision_service()
    assert service1 is service2
    assert service1.model is not None


def test_vision_model_inference_output():
    service = get_vision_service()
    dummy_img = Image.new("RGB", (224, 224), color=(120, 150, 180))

    embedding, predicted_room, confidence, class_probs = service.process_image(dummy_img)

    # 1. Validate embedding dimensions
    assert isinstance(embedding, list)
    assert len(embedding) == 512

    # 2. Validate L2 normalization (norm should be ~1.0)
    emb_arr = np.array(embedding, dtype=np.float32)
    norm = np.linalg.norm(emb_arr)
    assert pytest.approx(norm, rel=1e-3) == 1.0

    # 3. Validate predicted room category
    assert predicted_room in ROOM_CLASSES

    # 4. Validate confidence score bounds
    assert 0.0 <= confidence <= 1.0

    # 5. Validate class probabilities
    assert set(class_probs.keys()) == set(ROOM_CLASSES)
    assert all(0.0 <= prob <= 1.0 for prob in class_probs.values())
