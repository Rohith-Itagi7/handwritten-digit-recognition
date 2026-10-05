"""Train a simple CNN on the built-in Keras MNIST dataset."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential

from common import MODEL_PATH, PROJECT_ROOT


NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
EPOCHS = 8
BATCH_SIZE = 128


def create_model() -> Sequential:
    """Build a compact CNN that is easy to follow."""
    model = Sequential(
        [
            Conv2D(32, kernel_size=3, activation="relu", input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=2),
            Conv2D(64, kernel_size=3, activation="relu"),
            MaxPooling2D(pool_size=2),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_sample_digits(images: np.ndarray, labels: np.ndarray) -> None:
    """Save a small grid of training images and their labels."""
    figure, axes = plt.subplots(2, 5, figsize=(9, 4))
    for axis, image, label in zip(axes.flat, images[:10], labels[:10]):
        axis.imshow(image, cmap="gray")
        axis.set_title(f"Label: {label}")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(NOTEBOOKS_DIR / "sample_digits.png", dpi=150)
    plt.close(figure)


def save_training_graphs(history) -> None:
    """Save accuracy and loss plots for training and validation."""
    for metric, title, filename in (
        ("accuracy", "Training and Validation Accuracy", "training_accuracy.png"),
        ("loss", "Training and Validation Loss", "training_loss.png"),
    ):
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(history.history[metric], label="Training")
        axis.plot(history.history[f"val_{metric}"], label="Validation")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric.capitalize())
        axis.legend()
        figure.tight_layout()
        figure.savefig(NOTEBOOKS_DIR / filename, dpi=150)
        plt.close(figure)


def main() -> None:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Scale pixel values to 0-1 and add the single grayscale channel.
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0
    train_images = train_images[..., np.newaxis]
    test_images = test_images[..., np.newaxis]

    print(f"Training shape: {train_images.shape}")
    print(f"Testing shape: {test_images.shape}")
    print(f"Number of classes: {len(np.unique(train_labels))}")
    save_sample_digits(train_images, train_labels)

    model = create_model()
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
    )
    history = model.fit(
        train_images,
        train_labels,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
    )

    model.save(MODEL_PATH)
    save_training_graphs(history)
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Test images are ready for evaluation: {test_images.shape}")


if __name__ == "__main__":
    main()
