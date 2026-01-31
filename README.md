# 🍎🍌 Fruit Image Classifier — Deep Learning + TFLite + Streamlit

A lightweight, fast *fruit image classification app* built using *TensorFlow Lite* and deployed on *Hugging Face Spaces* with *Streamlit*.

Upload an image of a fruit, and the model predicts whether it's:

- 🍎 Apple Red  
- 🍌 Banana  
- 🍇 Grape White  
- 🥭 Mango

---

## 🚀 Live Demo

Click the badge to open the app:

[![Open in Spaces](https://huggingface.co/spaces/nandabr/fruit-image-classifier/badge.svg)](https://huggingface.co/spaces/nandabr/fruit-image-classifier)

---

## 🧠 Model Overview

This project uses:

- TensorFlow / Keras  
- Custom fruit dataset  
- Preprocessing: resizing + normalization  
- Exported to .tflite for fast inference  
- Deployed with Streamlit UI  

---

## 🏗 Tech Stack

| Component | Technology |
|----------|------------|
| Training | TensorFlow / Keras |
| Inference | TensorFlow Lite |
| UI | Streamlit |
| Deployment | Hugging Face Spaces |
| Language | Python |

---

## 📂 Project Structure


📦 fruit-image-classifier
 ├── app.py               # Streamlit app
 ├── fruit_model.tflite   # TFLite model
 ├── requirements.txt     # Dependencies
 ├── space.yaml           # HuggingFace runtime config
 └── README.md            # Documentation


---

## ▶️ Run Locally

### 1️⃣ Clone Repo
bash
git clone https://github.com/your-username/your-repo.git


### 2️⃣ Install Dependencies
bash
pip install -r requirements.txt


### 3️⃣ Run Streamlit App
bash
streamlit run app.py


Then open:

http://localhost:8501


---

## 📦 Requirements


streamlit
tensorflow
numpy
Pillow


---

## 🧩 How the Model Works

1. User uploads an image  
2. Image resized to *100×100*  
3. Normalized to 0–1  
4. Passed through TFLite interpreter  
5. Highest probability = final prediction  

---

## 🔮 Future Enhancements

- Add more fruit categories  
- Improved dataset  
- Add camera input  
- Build Android app  
- Real-time fruit detection  

---

## 👩‍💻 Author

*Nanda B R*  
Machine Learning | Deep Learning | AI  
HuggingFace: https://huggingface.co/nandabr  

---

## 📜 License

Open-source under MIT License.
