"""Shared paths, model loading, and image preprocessing."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "mnist_cnn.keras"


def load_trained_model():
    """Load the trained model or raise an actionable error."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Train it first by running: python src/train.py"
        )
    return load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Prepare an uploaded digit image as a normalized 28x28 CNN input."""
    grayscale = ImageOps.exif_transpose(image).convert("L")
    pixels = np.asarray(grayscale)

    # MNIST has bright digits on a dark background; invert common paper scans.
    corners = np.array(
        [pixels[0, 0], pixels[0, -1], pixels[-1, 0], pixels[-1, -1]]
    )
    if float(corners.mean()) > 127:
        grayscale = ImageOps.invert(grayscale)

    resized = grayscale.resize((28, 28), Image.Resampling.LANCZOS)
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    model_input = image_array.reshape(1, 28, 28, 1)
    return image_array, model_input
