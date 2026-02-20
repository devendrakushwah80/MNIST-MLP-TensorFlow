# 🧠 MNIST Digit Classification using Deep MLP (TensorFlow)

This project implements a Deep Multi-Layer Perceptron (MLP) model using TensorFlow to classify handwritten digits from the MNIST dataset.

---

## 📌 Project Overview

- Dataset: MNIST (Handwritten digits 0–9)
- Framework: TensorFlow / Keras
- Model Type: Deep Neural Network (MLP)
- Input Shape: 784 (28x28 flattened)
- Output Classes: 10
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy

---

## 📂 Dataset

The MNIST dataset contains:
- 60,000 training images
- 10,000 testing images
- Grayscale images (28x28 pixels)

Loaded using:

```python
tf.keras.datasets.mnist.load_data()
```
🔄 Data Preprocessing

Reshaped images from (28,28) → (784,)

Normalized pixel values (0–255 → 0–1)

x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0
🏗 Model Architecture
Layer	Units	Activation
Dense	256	ReLU
Dense	128	ReLU
Dense	64	ReLU
Dense	10	Softmax
⚙ Model Compilation
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
🏋 Training

Epochs: 10

Batch Size: 32

Validation Split: 20%

📊 Evaluation Metrics

Accuracy

Classification Report (Precision, Recall, F1-score)

Confusion Matrix (Heatmap)

Training vs Validation Accuracy Graph

📈 Results

High training and validation accuracy

Clear digit classification with minimal misclassification

Strong performance without using CNN
📦 Requirements

Install dependencies:

pip install -r requirements.txt
▶ How to Run

Clone the repository

Install requirements

Open Jupyter Notebook

Run MLP_Tf.ipynb

🎯 Future Improvements

Add Dropout layers

Add EarlyStopping

Convert to CNN for better performance

Add model saving & loading

Deploy with Streamlit

👨‍💻 Author

Devendra Kushwah
