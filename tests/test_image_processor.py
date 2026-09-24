import io
import pytest
import numpy as np
from PIL import Image
from app.services.image_processor import ImageProcessor


def test_load_image_from_bytes():
    # Create test image in memory
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    loaded = ImageProcessor.load_image_from_bytes(raw_bytes)
    assert loaded.mode == "RGB"
    assert loaded.size == (100, 100)


def test_preprocess_for_model():
    img = Image.new("RGB", (500, 300), color=(100, 150, 200))
    preprocessed = ImageProcessor.preprocess_for_model(img, target_size=(224, 224))

    assert preprocessed.shape == (224, 224, 3)
    assert preprocessed.dtype == np.float32
    assert 0.0 <= preprocessed.min() <= preprocessed.max() <= 1.0


def test_calculate_image_metrics():
    img = Image.new("RGB", (320, 240), color=(120, 120, 120))
    metrics = ImageProcessor.calculate_image_metrics(img)

    assert metrics["width"] == 320
    assert metrics["height"] == 240
    assert metrics["channels"] == 3
    assert "shannon_entropy" in metrics
    assert "contrast_metric" in metrics
    assert isinstance(metrics["shannon_entropy"], float)
    assert isinstance(metrics["contrast_metric"], float)
