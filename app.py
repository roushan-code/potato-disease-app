from pathlib import Path
from io import StringIO
import csv

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "potato_disease_efficientnetb2.keras"
IMAGE_SIZE = 500
CLASS_NAMES = [
    "Bacteria",
    "Fungi",
    "Healthy",
    "Nematode",
    "Pest",
    "Phytopthora",
    "Virus",
]


st.set_page_config(
    page_title="Potato Leaf Health Check",
    page_icon="🌿",
    layout="centered",
)

st.title("Potato Leaf Health Check")
st.write("Upload a clear potato-leaf image to get the model's disease prediction.")
st.caption("The result is an AI estimate, not a replacement for expert diagnosis.")


@st.cache_resource(show_spinner="Loading the EfficientNetB2 model...")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Place potato_disease_efficientnetb2.keras beside app.py."
        )
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

uploaded_file = st.file_uploader(
    "Choose one or more leaf images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if not uploaded_file:
    st.info("Choose one or more images to begin.")
    st.stop()

images = []
for uploaded_image in uploaded_file:
    try:
        image = ImageOps.exif_transpose(Image.open(uploaded_image)).convert("RGB")
        images.append((uploaded_image.name, image))
    except Exception:
        st.warning(f"Could not read {uploaded_image.name}; it was skipped.")

if not images:
    st.error("None of the uploaded files could be read as images.")
    st.stop()

st.write(f"Ready to analyze {len(images)} image(s).")

preview_columns = st.columns(min(len(images), 4))
for image_index, (image_name, image) in enumerate(images):
    with preview_columns[image_index % len(preview_columns)]:
        st.image(image, caption=image_name, use_container_width=True)

if st.button("Analyze images", type="primary", use_container_width=True):
    with st.spinner("Analyzing image..."):
        image_arrays = [
            np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32)
            for _, image in images
        ]
        image_batch = np.stack(image_arrays)

        # Predict in small chunks so many 500 x 500 images do not fill GPU memory.
        probabilities_by_image = []
        for start in range(0, len(image_batch), 8):
            probabilities_by_image.extend(
                model.predict(image_batch[start:start + 8], verbose=0)
            )

    results = []
    st.subheader("Predictions")
    for (image_name, image), probabilities in zip(images, probabilities_by_image):
        top_indices = np.argsort(probabilities)[::-1][:3]
        predicted_index = int(top_indices[0])
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(probabilities[predicted_index])

        results.append({
            "filename": image_name,
            "prediction": predicted_class,
            "confidence": f"{confidence:.2%}",
            "top_2": CLASS_NAMES[int(top_indices[1])],
            "top_3": CLASS_NAMES[int(top_indices[2])],
        })

        with st.expander(f"{image_name}: {predicted_class} ({confidence:.1%})"):
            st.image(image, caption=image_name, use_container_width=True)
            for index in top_indices:
                st.write(f"**{CLASS_NAMES[index]}**: {probabilities[index]:.1%}")
                st.progress(float(probabilities[index]))

            if confidence < 0.60:
                st.warning("The model is uncertain. Try a clearer image with the leaf filling most of the frame.")

    csv_buffer = StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
    st.download_button(
        "Download results CSV",
        data=csv_buffer.getvalue(),
        file_name="potato_disease_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
