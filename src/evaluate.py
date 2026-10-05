"""Evaluate the saved CNN on the MNIST test dataset."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.datasets import mnist

from common import PROJECT_ROOT, load_trained_model


def main() -> None:
    notebooks_dir = PROJECT_ROOT / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    del train_images, train_labels
    test_images = test_images.astype("float32") / 255.0
    test_images = test_images[..., np.newaxis]

    model = load_trained_model()
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    probabilities = model.predict(test_images, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4%}")
    print("\nClassification report:")
    print(
        classification_report(
            test_labels,
            predictions,
            labels=np.arange(10),
            zero_division=0,
        )
    )

    matrix = confusion_matrix(test_labels, predictions, labels=np.arange(10))
    print("Confusion matrix:")
    print(matrix)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=np.arange(10),
    )
    figure, axis = plt.subplots(figsize=(8, 8))
    display.plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title("MNIST Confusion Matrix")
    figure.tight_layout()
    figure.savefig(notebooks_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)
    print(f"Saved confusion matrix to: {notebooks_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
