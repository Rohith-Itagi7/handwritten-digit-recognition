# Handwritten Digit Recognition using Deep Learning

## Overview

This beginner-friendly project trains a Convolutional Neural Network (CNN) to recognize handwritten digits. It includes scripts to train and evaluate the model, a command-line prediction tool, and a small Streamlit web app.

## Objective

The goal is to recognize handwritten digits from 0 to 9 using a CNN trained on the MNIST dataset.

## Dataset

MNIST is a collection of handwritten digit images commonly used to learn image classification. Keras provides the dataset directly:

- 60,000 training images
- 10,000 test images
- 28 x 28 grayscale pixels per image
- 10 classes: digits 0 through 9

Pixel values are normalized from 0-255 to 0-1 before training.

## Technologies Used

- Python 3.10+
- TensorFlow / Keras
- Convolutional Neural Network (CNN)
- NumPy
- Matplotlib
- scikit-learn
- Pillow
- Streamlit

## Project Architecture

```text
MNIST Dataset
      ↓
Preprocessing
      ↓
CNN Model
      ↓
Training
      ↓
Evaluation
      ↓
Saved Model
      ↓
Streamlit App
      ↓
Digit Prediction
```

## CNN Architecture

1. **Convolution + ReLU:** learns visual patterns such as edges and curves.
2. **Max pooling:** reduces the feature-map size while retaining useful patterns.
3. **Second convolution + max pooling:** learns more complex digit features.
4. **Flatten:** turns the feature maps into a one-dimensional list.
5. **Dense + ReLU:** combines the learned features to make a prediction.
6. **Dropout:** reduces over-reliance on individual neurons during training.
7. **Dense with 10 softmax outputs:** returns a probability for each digit.

The model uses the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as its training metric.

## Results

After training and evaluating this project, the model achieved **98.92% test accuracy** on the 10,000-image MNIST test set (test loss: **0.0324**). Results may vary slightly with a different environment or training run.

## How to Run

Run these commands from the project root in Windows PowerShell or Command Prompt. Python 3.10 or newer is recommended.

```bat
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Train the model (downloads MNIST the first time):

```bat
python src/train.py
```

Evaluate the saved model:

```bat
python src/evaluate.py
```

Start the web application:

```bat
streamlit run app.py
```

The training script saves the model to `models/mnist_cnn.keras` and the sample and training plots to `notebooks/`. Evaluation saves the confusion matrix there as well.

### Command-line prediction

After training, pass the path to a PNG or JPG image containing a digit:

```bat
python src/predict.py path\to\digit.png
```

The script prints the predicted digit and the model's confidence.

## Example Prediction

In the Streamlit app, upload a clear image containing a single handwritten digit. The app displays the uploaded and preprocessed images, then shows the predicted digit and its confidence.

## Future Improvements

- Draw a digit directly on a canvas.
- Improve the CNN architecture.
- Add data augmentation.
- Deploy the application.
- Add prediction history.
