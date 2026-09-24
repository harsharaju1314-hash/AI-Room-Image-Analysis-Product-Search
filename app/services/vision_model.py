import logging
from typing import Tuple, Dict, List
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

logger = logging.getLogger(__name__)

ROOM_CLASSES = ["living_room", "bedroom", "kitchen", "bathroom"]

# Predefined ImageNet indices correlated with our target interior categories
# Allows realistic zero-training semantic aggregation from pretrained weights
IMAGENET_CATEGORY_MAPPING = {
    "living_room": [
        831, 861, 765, 849, 786, 681, 770, 779, 908  # studio couch, sofa, rocking chair, spotlight, table lamp, pillow
    ],
    "bedroom": [
        559, 810, 560, 483, 768, 623, 725  # four-poster bed, wardrobe, cradle, quilt, folding chair, lampshade
    ],
    "kitchen": [
        760, 659, 706, 524, 764, 459, 574  # refrigerator, microwave, espresso maker, dishwasher, toaster bottle
    ],
    "bathroom": [
        435, 898, 897, 856, 859, 883  # bathtub, washbasin, toilet seat, swab, syringe, towel dispenser
    ]
}


class VisionModelService:
    """
    Computer Vision Service powered by PyTorch ResNet-18.
    Provides:
    1. 512-dimensional visual embedding extraction (L2-normalized).
    2. Room classification across target categories (living_room, bedroom, kitchen, bathroom).
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing VisionModelService on device: {self.device}")

        # Load pretrained ResNet-18
        weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weights)
        self.model.eval()
        self.model.to(self.device)

        # Hook / feature extractor for 512-dim avgpool embedding
        self._embedding_hook = None
        self._latest_embedding = None
        self._register_hook()

        # Standard PyTorch normalization transform for ResNet
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _register_hook(self):
        """Registers a forward hook on the avgpool layer to intercept 512-dim features."""
        def hook(module, input, output):
            # output shape: (batch_size, 512, 1, 1) -> flatten to (batch_size, 512)
            flattened = torch.flatten(output, 1)
            # L2 normalize
            normalized = torch.nn.functional.normalize(flattened, p=2, dim=1)
            self._latest_embedding = normalized.detach().cpu().numpy()

        self.model.avgpool.register_forward_hook(hook)

    @torch.no_grad()
    def process_image(self, image: Image.Image) -> Tuple[List[float], str, float, Dict[str, float]]:
        """
        Runs full forward pass on image.
        Returns:
        - embedding: List[float] (512-dim L2-normalized vector)
        - room_type: str (predicted category)
        - confidence: float (0.0 to 1.0)
        - class_probabilities: Dict[str, float]
        """
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Forward pass (triggers avgpool hook for embedding)
        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        embedding = self._latest_embedding[0].tolist()

        # Calculate category scores from mapped semantic representations
        raw_scores = {}
        for room, indices in IMAGENET_CATEGORY_MAPPING.items():
            score = float(np.sum(probabilities[indices]))
            raw_scores[room] = score

        # Add small baseline prior for numerical stability
        total_raw = sum(raw_scores.values()) + 1e-6
        normalized_scores = {k: v / total_raw for k, v in raw_scores.items()}

        # If image has weak specific cues, apply soft entropy smoothing
        if max(normalized_scores.values()) < 0.35:
            # Distribute softly while keeping highest ranked
            top_class = max(normalized_scores, key=normalized_scores.get)
            normalized_scores = {k: 0.20 for k in ROOM_CLASSES}
            normalized_scores[top_class] = 0.40

        # Sort and pick top prediction
        predicted_room = max(normalized_scores, key=normalized_scores.get)
        confidence = float(normalized_scores[predicted_room])

        formatted_probs = {k: round(float(v), 4) for k, v in normalized_scores.items()}

        return embedding, predicted_room, round(confidence, 4), formatted_probs


# Singleton instance
_vision_service_instance = None


def get_vision_service() -> VisionModelService:
    global _vision_service_instance
    if _vision_service_instance is None:
        _vision_service_instance = VisionModelService()
    return _vision_service_instance
