import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import tensorflow.lite as tflite
import numpy as np
from PIL import Image

# -------------------------
# LOAD TFLITE MODEL
# -------------------------
MODEL_PATH = "models/fruit_model.tflite"

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class names must match training order
class_names = ['Apple Red 1', 'Banana 1', 'Grape White 1', 'Mango 1']


# -------------------------
# PREDICTION FUNCTION
# -------------------------
def predict_image(image):
    img = image.resize((100, 100))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    pred = interpreter.get_tensor(output_details[0]['index'])
    return pred


# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="Fruit Image Classifier", layout="centered")

st.title("🍎 Fruit Image Classifier")
st.write("Upload an image of a fruit, and the model will predict which fruit it is.")

uploaded_file = st.file_uploader("Choose a fruit image:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Predicting...")
    pred = predict_image(image)

    class_index = np.argmax(pred)
    confidence = np.max(pred)

    st.success(f"Predicted Fruit: *{class_names[class_index]}*")
    st.info(f"Confidence: *{confidence * 100:.2f}%*")
