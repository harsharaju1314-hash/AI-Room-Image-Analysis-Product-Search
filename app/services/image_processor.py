import io
from typing import Tuple, Dict, Any
import numpy as np
import cv2
from PIL import Image
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops


class ImageProcessor:
    """
    Image preprocessing and visual quality analysis service.
    Integrates Pillow (I/O, formatting), OpenCV (resizing, color conversions),
    and scikit-image (entropy & texture contrast extraction).
    """

    @staticmethod
    def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
        """
        Loads an image from raw bytes using Pillow and converts to RGB.
        """
        image_stream = io.BytesIO(image_bytes)
        image = Image.open(image_stream)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    @staticmethod
    def preprocess_for_model(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocesses PIL image using OpenCV for neural network input:
        1. Converts PIL RGB image to NumPy array.
        2. Resizes to target dimension (224x224) using OpenCV INTER_AREA / INTER_LINEAR.
        3. Normalizes pixel values to [0.0, 1.0] range.
        """
        np_img = np.array(image)
        # OpenCV resize
        resized = cv2.resize(np_img, target_size, interpolation=cv2.INTER_LINEAR)
        # Normalize to float32 [0.0, 1.0]
        normalized = resized.astype(np.float32) / 255.0
        return normalized

    @staticmethod
    def calculate_image_metrics(image: Image.Image) -> Dict[str, Any]:
        """
        Computes meaningful image characteristics:
        - Image dimensions (width, height, channels)
        - Shannon Entropy (visual complexity / detail richness) via scikit-image
        - GLCM Contrast (lighting & edge texture contrast) via scikit-image
        """
        np_img = np.array(image)
        height, width = np_img.shape[:2]
        channels = np_img.shape[2] if len(np_img.shape) > 2 else 1

        # Convert RGB to Grayscale for texture & contrast analysis
        gray_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

        # 1. Shannon Entropy via scikit-image
        # Measures visual information density / clutter level
        entropy_val = float(shannon_entropy(gray_img))

        # 2. GLCM Texture Contrast via scikit-image
        # Downscale grayscale image for fast GLCM computation
        small_gray = cv2.resize(gray_img, (128, 128), interpolation=cv2.INTER_AREA)
        glcm = graycomatrix(
            small_gray,
            distances=[1],
            angles=[0],
            levels=256,
            symmetric=True,
            normed=True
        )
        contrast_val = float(graycoprops(glcm, "contrast")[0, 0])

        return {
            "width": width,
            "height": height,
            "channels": channels,
            "shannon_entropy": round(entropy_val, 4),
            "contrast_metric": round(contrast_val, 4)
        }
