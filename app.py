"""Streamlit interface for handwritten digit prediction."""

import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.common import load_trained_model, preprocess_image


st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="🔢")
st.title("Handwritten Digit Recognition")
st.write("Upload an image of one handwritten digit to see the model's prediction.")

uploaded_file = st.file_uploader(
    "Choose a digit image",
    type=["png", "jpg", "jpeg"],
    help="For best results, upload one clear digit with minimal background.",
)


@st.cache_resource(max_entries=1)
def get_model():
    """Load the trained model once and reuse it between Streamlit reruns."""
    return load_trained_model()


if uploaded_file is not None:
    try:
        with Image.open(uploaded_file) as source_image:
            uploaded_image = source_image.copy()
        processed_image, model_input = preprocess_image(uploaded_image)
    except (UnidentifiedImageError, OSError) as error:
        st.error(f"Could not read the uploaded image: {error}")
    else:
        left_column, right_column = st.columns(2)
        with left_column:
            st.subheader("Uploaded Image")
            st.image(
                uploaded_image,
                width="stretch",
                alt="Uploaded handwritten digit image",
            )
        with right_column:
            st.subheader("Preprocessed Image")
            st.image(
                processed_image,
                clamp=True,
                width="stretch",
                alt="Grayscale 28 by 28 image prepared for the CNN",
            )

        try:
            model = get_model()
        except FileNotFoundError as error:
            st.error(str(error))
        else:
            probabilities = model.predict(model_input, verbose=0)[0]
            predicted_digit = int(probabilities.argmax())
            confidence = float(probabilities[predicted_digit])
            st.success(f"Predicted digit: {predicted_digit}")
            st.metric("Confidence", f"{confidence:.1%}")
