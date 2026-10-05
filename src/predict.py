"""Predict the digit in a PNG or JPG image from the command line."""

import argparse
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from common import load_trained_model, preprocess_image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict the digit shown in an image."
    )
    parser.add_argument("image_path", type=Path, help="Path to a digit image")
    args = parser.parse_args()

    if not args.image_path.is_file():
        parser.error(f"Image file does not exist: {args.image_path}")

    try:
        with Image.open(args.image_path) as image:
            _, model_input = preprocess_image(image)
    except (UnidentifiedImageError, OSError) as error:
        parser.error(f"Could not read image '{args.image_path}': {error}")

    try:
        model = load_trained_model()
    except FileNotFoundError as error:
        parser.error(str(error))

    probabilities = model.predict(model_input, verbose=0)[0]
    predicted_digit = int(probabilities.argmax())
    confidence = float(probabilities[predicted_digit])
    print(f"Predicted digit: {predicted_digit}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
