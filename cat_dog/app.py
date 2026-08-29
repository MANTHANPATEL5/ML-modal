import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import numpy as np

# ==========================================================
# PAGE SETTINGS
# ==========================================================

st.set_page_config(
    page_title="Cat vs Dog Classification",
    page_icon="🐱",
    layout="centered"
)

st.title("🐱 🐶 Cat vs Dog Image Classification")

st.write(
    "Upload an image and the model will predict whether it is a Cat or Dog."
)

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cat_dog_model.keras")

model = load_model()

# ==========================================================
# USER INPUT
# ==========================================================

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

# ==========================================================
# PREDICTION
# ==========================================================

if uploaded_file is not None:

    try:

        # Read uploaded file
        image_bytes = uploaded_file.getvalue()

        # Open image
        image = Image.open(BytesIO(image_bytes))

        # Convert to RGB
        image = image.convert("RGB")

        # Display image
        st.image(
            image,
            caption="Uploaded Image",
            width="stretch"
        )

        # Resize
        img = image.resize((160, 160))

        # Convert to NumPy
        img_array = np.array(img)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # ==================================================
        # PREDICTION
        # ==================================================

        prediction = model.predict(
            img_array,
            verbose=0
        )[0][0]

        # ==================================================
        # RESULT
        # ==================================================

        if prediction >= 0.5:

            result = "🐶 DOG"
            confidence = prediction * 100

        else:

            result = "🐱 CAT"
            confidence = (1 - prediction) * 100

        st.subheader("Prediction")

        if prediction >= 0.5:
            st.success(result)
        else:
            st.info(result)

        st.write(
            f"Confidence: **{confidence:.2f}%**"
        )

    except UnidentifiedImageError:

        st.error(
            "❌ Unable to read the image. "
            "Please upload a valid image."
        )

    except Exception as e:

        st.error(f"❌ Error: {e}")
