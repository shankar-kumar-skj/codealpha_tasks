# Handwritten Character Recognition
This project is a Handwritten Digit Recognition system built using Machine Learning and Deep Learning.
It recognizes handwritten digits (0–9) using the MNIST dataset and a trained neural network model.

# 📁 Project Structure

Handwritten_Character_Recognition/
│
├── app.ipynb
├── mnist_99_model.h5
└── mnist.npz


# 📄 File Description
    • app.ipynb
## Jupyter Notebook containing:
        ○ Data loading
        ○ Model training
        ○ Model testing
        ○ Prediction logic
    • mnist_99_model.h5
## Pre-trained deep learning model with ~99% accuracy on MNIST dataset.
    • mnist.npz
## MNIST dataset file containing handwritten digit images and labels.

# 🚀 How to Run the Project
## 1. Install Required Libraries
Make sure Python is installed, then install dependencies:

pip install numpy tensorflow keras matplotlib jupyter

## 2. Open the Jupyter Notebook

jupyter notebook
Open app.ipynb from the browser.

## 3. Run the Notebook
    • Run all cells step by step
    • The model will load and predict handwritten digits

# 🧠 Dataset Used
    • MNIST Dataset
        ○ 60,000 training images
        ○ 10,000 testing images
        ○ Image size: 28×28
        ○ Digits: 0 to 9

# ⚙️ Model Details
    • Neural Network / CNN based model
    • Trained using TensorFlow & Keras
    • Saved as .h5 file for reuse

# 📊 Output
    • Predicts the digit shown in the handwritten image
    • Displays accuracy and predictions visually

# 🎯 Use Cases
    • Learning Machine Learning & Deep Learning
    • Digit recognition systems
    • Academic projects
    • Beginner AI projects

# 📜 License
This project is for educational purposes.
