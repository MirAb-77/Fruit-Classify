# Fruit Classification Web App

## 🍎 Project Overview
This project is a **Fruit Classification Web App** that uses deep learning techniques to classify 24 types of fruits from images. The model is trained using **Convolutional Neural Networks (CNN)** with **Transfer Learning** applied from the **VGG16** model to improve performance.

---

## 🔑 Techniques Used
- Deep Learning with **TensorFlow and Keras**
- **Transfer Learning** (VGG16 Pre-trained Model)
- Data Augmentation
- Batch Normalization
- Dropout Regularization
- Image Preprocessing & Rescaling
- Model Training & Fine-Tuning
- Web App Development using **Streamlit**
- Gradient Background and Modern UI Design

---

## 📌 Dataset Used
### **Fruits-360 Dataset**
- URL: [Download Dataset](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/4yIRGlIpNfKEGJYMhZV52g/fruits-360-original-size.zip)
- **Training Images**: 6231 images belonging to 24 classes
- **Validation Images**: 3114 images belonging to 24 classes
- **Test Images**: 3110 images belonging to 24 classes

**Dataset Description:**
The dataset contains high-quality images of 24 different fruit classes, including Apple, Banana, Orange, Pineapple, Watermelon, Kiwi, Mango, and more.

---

## 🎯 Aim
To develop an interactive web application that can classify fruit images with high accuracy using a deep learning model.

### Objectives:
1. Build a Fruit Classification Model using **Transfer Learning (VGG16)**.
2. Train the model on the **Fruits-360 Dataset**.
3. Achieve high accuracy using **Data Augmentation** and **Fine-Tuning**.
4. Develop a User-Friendly Web Interface using **Streamlit**.
5. Provide predictions with confidence scores.

---

## 🔥 Model Architecture
The model uses **VGG16** as the base model with the following architecture:

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout

# Load VGG16 with pre-trained weights
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Build the model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(24, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---

## 🔍 Training and Testing
### Training Configuration
- Batch Size: 32
- Image Size: (64, 64)
- Epochs: 30
- Optimizer: Adam
- Loss Function: Categorical Cross-Entropy

### Accuracy Graphs:
#### Training and Validation Accuracy
![Accuracies](<img width="302" alt="image" src="https://github.com/user-attachments/assets/d10bc160-626c-477d-8fb0-746bbcf331e3" />
)

#### Training and Validation Loss
(<img width="345" alt="image" src="https://github.com/user-attachments/assets/856197a1-f55e-4d04-b79d-7618ba2419ac" />
)

---

## 📌 Web App Design
The web app is developed using **Streamlit** with a modern gradient UI featuring:
- Upload Image Functionality
- Interactive Predict Button
- Hover Effects
- Dynamic Results with Confidence Score
- Sidebar Instructions
- Custom CSS Styling

### Web App Preview:
![Web App](<img width="431" alt="image" src="https://github.com/user-attachments/assets/71c7b03d-ec1e-4bcd-a11f-888cae737065" />
)

---

## 🎯 Results
| Metric                 | Value       |
|-----------------------|-------------|
| Test Accuracy         | **86.8%**   |

### Sample Predictions:
| Fruit Image           | Predicted Label | Confidence Score |
|-----------------------|----------------|-----------------|
| 🍎 Apple              | Apple         | 89.1%          |
| 🍌 Banana             | Banana        | 78.3%          |
| 🍉 Watermelon         | Watermelon    | 82.6%          |

---

## 🎯 How to Run the App
1. Install the required libraries:
```bash
pip install streamlit tensorflow pillow numpy
```
2. Clone the Repository:
```bash
git clone https://github.com/MirAb-77/Fruit-Classification-Web-App.git
```
3. Run the App:
```bash
streamlit run app.py
```

---

## 🌐 Web App Demo
👉 [Fruit Classification Web App](https://github.com/MirAb-77/Fruit-Classification-Web-App)

---

## 💪 Future Improvements
- Add more fruit categories
- Improve Model Accuracy
- Deploy Web App on Cloud Platforms
- Add **Multiple Image Upload** Feature
- Include **Bounding Box Detection** for multiple fruits in one image

---

## 📌 Author
**Abdullah Imran**  
💼 Aspiring Data Scientist & ML Developer

Connect with me:
- [LinkedIn](https://www.linkedin.com/in/abdullah-mir-211658230/)
- [GitHub](https://github.com/MirAb-77)
- [Portfolio](#)

---

## 🛑 License
This project is open-source and free to use for educational purposes.

---

### If you like this project, don't forget to ⭐ star the repository!

