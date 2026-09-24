import logging
import os
from typing import Tuple, Dict, List
import numpy as np
from PIL import Image

# Suppress noisy TensorFlow info/warning logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model

logger = logging.getLogger(__name__)

ROOM_CLASSES = ["living_room", "bedroom", "kitchen", "bathroom"]

# Predefined ImageNet indices correlated with our target interior categories
IMAGENET_CATEGORY_MAPPING = {
    "living_room": [
        831, 861, 765, 849, 786, 681, 770, 779, 908  # studio couch, sofa, rocking chair, spotlight, table lamp, pillow
    ],
    "bedroom": [
        559, 810, 560, 483, 768, 623, 725  # four-poster bed, wardrobe, cradle, quilt, folding chair, lampshade
    ],
    "kitchen": [
        760, 659, 706, 524, 764, 459, 574  # refrigerator, microwave, espresso maker, dishwasher, toaster
    ],
    "bathroom": [
        435, 898, 897, 856, 859, 883  # bathtub, washbasin, toilet seat, towel dispenser
    ]
}


class VisionModelService:
    """
    Computer Vision Service powered by TensorFlow / Keras MobileNetV2.
    Provides:
    1. 512-dimensional visual embedding extraction (L2-normalized).
    2. Room classification across target categories (living_room, bedroom, kitchen, bathroom).
    """

    def __init__(self):
        logger.info("Initializing VisionModelService with TensorFlow MobileNetV2...")
        
        # Load base pretrained model
        self.base_model = MobileNetV2(weights="imagenet", include_top=True)
        self.model = self.base_model  # alias for health check

        # Feature extractor model intercepting global average pooling representation (1280-dim)
        gap_layer = self.base_model.get_layer("global_average_pooling2d")
        self.feature_extractor = Model(inputs=self.base_model.input, outputs=gap_layer.output)

        # 512-dim projection matrix (deterministic projection for consistent 512-dim embeddings)
        np.random.seed(42)
        proj = np.random.randn(1280, 512).astype(np.float32)
        self.projection_matrix = proj / np.linalg.norm(proj, axis=0)

    def process_image(self, image: Image.Image) -> Tuple[List[float], str, float, Dict[str, float]]:
        """
        Runs forward inference on image using TensorFlow.
        Returns:
        - embedding: List[float] (512-dim L2-normalized vector)
        - room_type: str (predicted category)
        - confidence: float (0.0 to 1.0)
        - class_probabilities: Dict[str, float]
        """
        # Resize to 224x224 and prepare array
        resized_img = image.resize((224, 224), Image.Resampling.BILINEAR)
        img_array = np.array(resized_img, dtype=np.float32)
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]

        # Expand dims for batch: (1, 224, 224, 3)
        input_tensor = np.expand_dims(img_array, axis=0)
        preprocessed = preprocess_input(input_tensor)

        # 1. Extract 1280-dim feature representation
        raw_features = self.feature_extractor(preprocessed, training=False).numpy()[0]

        # 2. Project to 512-dim embedding and L2 normalize
        projected_512 = np.dot(raw_features, self.projection_matrix)
        norm = np.linalg.norm(projected_512)
        if norm > 0:
            embedding_512 = (projected_512 / norm).tolist()
        else:
            embedding_512 = projected_512.tolist()

        # 3. Classify Room Type from full model output probabilities
        predictions = self.base_model(preprocessed, training=False).numpy()[0]

        raw_scores = {}
        for room, indices in IMAGENET_CATEGORY_MAPPING.items():
            valid_indices = [i for i in indices if i < len(predictions)]
            score = float(np.sum(predictions[valid_indices]))
            raw_scores[room] = score

        total_raw = sum(raw_scores.values()) + 1e-6
        normalized_scores = {k: v / total_raw for k, v in raw_scores.items()}

        if max(normalized_scores.values()) < 0.35:
            top_class = max(normalized_scores, key=normalized_scores.get)
            normalized_scores = {k: 0.20 for k in ROOM_CLASSES}
            normalized_scores[top_class] = 0.40

        predicted_room = max(normalized_scores, key=normalized_scores.get)
        confidence = float(normalized_scores[predicted_room])

        formatted_probs = {k: round(float(v), 4) for k, v in normalized_scores.items()}

        return embedding_512, predicted_room, round(confidence, 4), formatted_probs


# Singleton instance
_vision_service_instance = None


def get_vision_service() -> VisionModelService:
    global _vision_service_instance
    if _vision_service_instance is None:
        _vision_service_instance = VisionModelService()
    return _vision_service_instance
