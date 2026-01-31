import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import streamlit as st
import tensorflow.lite as tflite
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
MODEL_PATH = "models/fruit_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class names (MUST be same order as training)
class_names = ['Apple Red 1', 'Banana 1', 'Grape White 1', 'Mango 1']

st.set_page_config(page_title="Fruit Classifier", layout="centered")

st.title("🍎 Fruit Image Classifier")
st.write("Upload an image of a fruit and the model will predict it.")

uploaded_file = st.file_uploader(
    "Choose a fruit image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Show image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(100, 100))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    if st.button("Predict Fruit"):
        with st.spinner("Predicting..."):
            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions)]
            confidence = np.max(predictions) * 100

        st.success(f"### 🥭 Predicted Fruit: *{predicted_class}*")

        st.info(f"Confidence: *{confidence:.2f}%*")
