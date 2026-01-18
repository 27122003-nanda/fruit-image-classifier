import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model("models/fruit_model.h5")

# Classes used during training
class_names = ["Apple Red 1", "Banana 1", "Grape White 1", "Mango 1"]

def predict_fruit(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(100, 100))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    return class_names[class_index]

# Ask user for file
img_path = input("Enter image path: ")
print("\nPredicting...")

result = predict_fruit(img_path)
print("\nPredicted Fruit:", result)