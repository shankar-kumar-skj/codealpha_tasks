# 💳 Credit Score Prediction System
A Machine Learning–based project that predicts whether a customer will default on their credit card payment next month using historical financial data.
The project also includes a Streamlit web application for real-time predictions.

# 📁 Project Structure

Credit_Score_Model/
│
├── app.ipynb
├── app.py
├── default of credit card clients.xls
├── default_of_credit_card_clients.csv
├── README.md


# 🎯 Project Objective
    • Predict credit default risk of customers
    • Analyze customer financial behavior
    • Build an end-to-end ML project (training → evaluation → deployment)
    • Suitable for Final Year Project / Internship / Portfolio


# 📊 Dataset Description
    • Dataset Name: Default of Credit Card Clients
    • Source: UCI Machine Learning Repository
    • Total Records: 30,000+
    • Target Variable:
## default payment next month
        ○ 0 → No default
        ○ 1 → Default
## Main Features
    • LIMIT_BAL – Credit limit
    • SEX, EDUCATION, MARRIAGE, AGE
    • PAY_0 to PAY_6 – Payment history
    • BILL_AMT1 to BILL_AMT6
    • PAY_AMT1 to PAY_AMT6


# 🧠 Machine Learning Workflow
    1. Data loading and cleaning
    2. Encoding categorical variables
    3. Feature engineering
    4. Feature scaling
    5. Handling class imbalance (SMOTE)
    6. Model training
    7. Model evaluation
    8. Deployment using Streamlit


# 🤖 Models Used
    • Random Forest Classifier
    • LightGBM Classifier
    • Voting Classifier (Ensemble Model)


# 📈 Model Performance (Approximate)
## Metric	Score
Accuracy	~80–81%
ROC-AUC	~0.76–0.77
Precision (Default)	~0.58
Recall (Default)	~0.45
### ⚠️ Note:
Achieving 98% accuracy is not realistic for real-world credit risk datasets due to noise, imbalance, and uncertainty.


# 📉 Visualizations Included
    • Confusion Matrix
    • ROC Curve
    • Feature Importance Plot
(All generated inside app.ipynb)


# 🖥️ Streamlit Web Application
Features:
    • User-friendly input form
    • Real-time prediction
    • Default probability output
    • Clear visual feedback


# ▶️ How to Run the Project
## 1️⃣ Install Dependencies

pip install pandas numpy scikit-learn lightgbm imbalanced-learn streamlit matplotlib seaborn joblib



## 2️⃣ Train the Model
Open and run:

app.ipynb

This will:
    • Train the ML model
    • Evaluate performance
    • Save the trained model


## 3️⃣ Run Streamlit App

streamlit run app.py

Open in browser:

http://localhost:8501



# ⚠️ Limitations
    • Dataset is not region-specific
    • Not suitable for real banking systems
    • Educational use only


# 🚀 Future Improvements
    • SHAP / LIME explainability
    • Deep learning models
    • REST API using FastAPI
    • Cloud deployment (AWS / GCP)


# 🎓 Best Use Cases
    • Final Year Academic Project
    • Internship Submission
    • Machine Learning Portfolio
    • GitHub Showcase


# 📌 Disclaimer
This project is for educational purposes only.
Do not use it for real financial or lending decisions.
