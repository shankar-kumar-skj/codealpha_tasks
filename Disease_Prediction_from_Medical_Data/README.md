# 🏥 Disease Prediction from Medical Data
This project predicts the possibility of diseases using patient medical data and Machine Learning classification algorithms.
It supports multiple medical datasets and automatically selects the best-performing model for each disease.

# 📌 Project Objective
To predict whether a patient has a disease based on:
    • Age
    • Symptoms
    • Medical measurements
    • Blood test results

# 🧠 Approach Used
    • Problem Type: Classification
    • Algorithms Used:
        ○ Logistic Regression
        ○ Support Vector Machine (SVM)
        ○ Random Forest
        ○ XGBoost
    • Best model is selected based on accuracy

# 📂 Project Structure

Disease_Prediction_from_Medical_Data/
│
├── datasets/
│   ├── heart.csv
│   ├── diabetes.csv
│   └── breast_cancer.csv
│
├── app.ipynb
├── heart_model.pkl
├── diabetes_model.pkl
├── cancer_model.pkl
└── README.md


# 📊 Datasets Used
All datasets are taken from the UCI Machine Learning Repository.
## 1️⃣ Heart Disease Dataset
    • Target Column: num
    • 0 → No disease
    • >0 → Disease present (converted to binary)
## 2️⃣ Diabetes Dataset
    • Target Column: Outcome
    • 0 → No diabetes
    • 1 → Diabetes
## 3️⃣ Breast Cancer Dataset
    • Target Column: diagnosis
    • B → Benign
    • M → Malignant
    • Dropped columns: id, Unnamed: 32

# ⚙️ Technologies Used
    • Python 3.10+
    • Pandas
    • NumPy
    • Scikit-learn
    • XGBoost
    • Jupyter Notebook

# 🚀 How to Run the Project
## Step 1: Install Dependencies
pip install numpy pandas scikit-learn xgboost

## Step 2: Open the Notebook
jupyter notebook app.ipynb

## Step 3: Run All Cells
    • Models will be trained
    • Best model for each dataset will be saved as .pkl

# 🧪 Model Output
After training, the following models are generated:
    • heart_model.pkl
    • diabetes_model.pkl
    • cancer_model.pkl
These models can be used to predict disease from new patient data.

# 🧠 Sample Prediction Logic

import pickle
model = pickle.load(open("heart_model.pkl", "rb"))
result = model.predict([[63,1,3,145,233,1,150,0,2.3,1]])
print("Disease Detected" if result[0] == 1 else "No Disease")

# 📈 Results
    • Heart Disease Accuracy: 83%
    • Diabetes Accuracy: 88%
    • Breast Cancer Accuracy: 97%
(Random Forest / XGBoost performed best)

# 🎯 Applications
    • Early disease detection
    • Medical decision support
    • Healthcare analytics
    • Academic & internship projects

# 📝 Conclusion
This project demonstrates how Machine Learning can be used to predict diseases from structured medical data.
By comparing multiple algorithms and selecting the best model, the system ensures reliable predictions.
