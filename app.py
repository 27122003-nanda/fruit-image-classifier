import streamlit as st
import numpy as np
from PIL import Image

# Load TFLite model
import tensorflow.lite as tflite

# Load trained model
MODEL_PATH = "fruit_model.tflite"
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class names (same order as training)
class_names = ['Apple Red 1', 'Banana 1', 'Grape White 1', 'Mango 1']

st.set_page_config(page_title="Fruit Classifier", layout="centered")

st.title("🍎 Fruit Image Classifier")
st.write("Upload an image of a fruit and the model will classify it.")

uploaded_file = st.file_uploader("Choose a fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((100, 100))
    st.image(image, caption="Uploaded Image", width=200)

    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"*Prediction:* {predicted_class}")
    st.info(f"*Confidence:* {confidence:.2f}%")
