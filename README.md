# 😷 Face Mask Detection using CNN  

A deep learning-based system to detect whether a person is wearing a mask or not in real time.  

---

## 🔹 Overview  
This project uses a Convolutional Neural Network (CNN) model along with OpenCV to detect faces and classify them as **With Mask 😷** or **Without Mask 🚫**.  

---

## ✅ Key Features  
- Real-time face detection using OpenCV 📸  
- Binary classification: With Mask / Without Mask 🛠️  
- Handles multiple faces simultaneously 👥   
- Easy to run on webcam or external video feed 🎥  

---

## 🔹 Tech Stack  
- 🟢 Python (NumPy, Pandas, Matplotlib)  
- 🟢 OpenCV (face detection & video processing)  
- 🟢 TensorFlow / Keras (CNN model for classification)  
- 🟢 Pickle (for class labels storage)  

---

## 🔹 Dataset  
The dataset consists of **images of people with and without masks**.  
- Classes: `with_mask`, `without_mask`  
- Images are preprocessed (resized, normalized) before training.  

---

## 🔹 Implementation Steps  
1️⃣ Load dataset and preprocess images  
2️⃣ Train CNN model on mask/no-mask data  
3️⃣ Save model & class labels  
4️⃣ Perform real-time face detection using OpenCV  
5️⃣ Classify each detected face using the trained CNN model  

---

## 🔹 Results & Accuracy  
📊 Model Accuracy: **~93%** (CNN)  

- High precision and recall for both classes  
- Works efficiently in real-time scenarios  

---

## 🔹 How to Use?  
1. Clone the repository  
2. Install required dependencies (`pip install -r requirements.txt`)  
3. Run the detection script:  

```bash
python realtime.py

---

## 🔹 **Future Improvements**
✅ Deploy as a Flask / Streamlit web app 🌐
✅ Extend dataset for better generalization 📂
✅ Optimize CNN architecture for faster inference ⚡
✅ Add mask type detection (cloth, surgical, N95) 🏷️
