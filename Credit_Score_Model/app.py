import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Load Trained Ensemble Model
# ===============================
ensemble_model = joblib.load('creditworthiness_model.pkl')

st.set_page_config(page_title="Creditworthiness Prediction", layout="wide")
st.title("Creditworthiness Prediction App")
st.write("Predict whether a customer will default next month.")

# ===============================
# User Input Section
# ===============================
st.sidebar.header("Customer Inputs")

def user_input_features():
    LIMIT_BAL = st.sidebar.number_input("Credit Limit (LIMIT_BAL)", min_value=0, value=50000)
    SEX = st.sidebar.selectbox("Sex", options=[1,2], format_func=lambda x: "Male" if x==1 else "Female")
    EDUCATION = st.sidebar.selectbox("Education", options=[1,2,3,4])
    MARRIAGE = st.sidebar.selectbox("Marriage Status", options=[1,2,3])
    AGE = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)

    PAY_0 = st.sidebar.number_input("PAY_0", value=0)
    PAY_2 = st.sidebar.number_input("PAY_2", value=0)
    PAY_3 = st.sidebar.number_input("PAY_3", value=0)
    PAY_4 = st.sidebar.number_input("PAY_4", value=0)
    PAY_5 = st.sidebar.number_input("PAY_5", value=0)
    PAY_6 = st.sidebar.number_input("PAY_6", value=0)

    BILL_AMT1 = st.sidebar.number_input("BILL_AMT1", value=0)
    BILL_AMT2 = st.sidebar.number_input("BILL_AMT2", value=0)
    BILL_AMT3 = st.sidebar.number_input("BILL_AMT3", value=0)
    BILL_AMT4 = st.sidebar.number_input("BILL_AMT4", value=0)
    BILL_AMT5 = st.sidebar.number_input("BILL_AMT5", value=0)
    BILL_AMT6 = st.sidebar.number_input("BILL_AMT6", value=0)

    PAY_AMT1 = st.sidebar.number_input("PAY_AMT1", value=0)
    PAY_AMT2 = st.sidebar.number_input("PAY_AMT2", value=0)
    PAY_AMT3 = st.sidebar.number_input("PAY_AMT3", value=0)
    PAY_AMT4 = st.sidebar.number_input("PAY_AMT4", value=0)
    PAY_AMT5 = st.sidebar.number_input("PAY_AMT5", value=0)
    PAY_AMT6 = st.sidebar.number_input("PAY_AMT6", value=0)

    data = {
        "LIMIT_BAL": LIMIT_BAL,
        "SEX": 0 if SEX==1 else 1,
        "EDUCATION": EDUCATION-1,
        "MARRIAGE": MARRIAGE-1,
        "AGE": AGE,
        "PAY_0": PAY_0,
        "PAY_2": PAY_2,
        "PAY_3": PAY_3,
        "PAY_4": PAY_4,
        "PAY_5": PAY_5,
        "PAY_6": PAY_6,
        "BILL_AMT1": BILL_AMT1,
        "BILL_AMT2": BILL_AMT2,
        "BILL_AMT3": BILL_AMT3,
        "BILL_AMT4": BILL_AMT4,
        "BILL_AMT5": BILL_AMT5,
        "BILL_AMT6": BILL_AMT6,
        "PAY_AMT1": PAY_AMT1,
        "PAY_AMT2": PAY_AMT2,
        "PAY_AMT3": PAY_AMT3,
        "PAY_AMT4": PAY_AMT4,
        "PAY_AMT5": PAY_AMT5,
        "PAY_AMT6": PAY_AMT6
    }

    # Extra engineered features for efficiency
    avg_bill = np.mean([BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6])
    avg_pay = np.mean([PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6])
    debt_ratio = avg_bill / (LIMIT_BAL+1e-5)
    payment_ratio = avg_pay / (avg_bill+1e-5)
    max_delay = max(PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6)
    num_delay_months = sum([1 if x>0 else 0 for x in [PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6]])
    bill_variance = np.var([BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6])
    pay_variance = np.var([PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6])
    recent_pay_ratio = PAY_AMT6 / (BILL_AMT6+1e-5)
    last_month_diff = BILL_AMT6 - BILL_AMT5

    # Add extra features
    extra_features = {
        "avg_bill": avg_bill,
        "avg_pay": avg_pay,
        "debt_ratio": debt_ratio,
        "payment_ratio": payment_ratio,
        "max_delay": max_delay,
        "num_delay_months": num_delay_months,
        "bill_variance": bill_variance,
        "pay_variance": pay_variance,
        "recent_pay_ratio": recent_pay_ratio,
        "last_month_diff": last_month_diff
    }

    data.update(extra_features)
    features_df = pd.DataFrame(data, index=[0])
    return features_df

input_df = user_input_features()

# ===============================
# Scaling (optional)
# ===============================
scaler = StandardScaler()
input_scaled = scaler.fit_transform(input_df)

# ===============================
# Prediction
# ===============================
if st.button("Predict"):
    prediction = ensemble_model.predict(input_scaled)[0]
    probability = ensemble_model.predict_proba(input_scaled)[0][1]

    if prediction == 0:
        st.success(f"✅ Customer is likely to **repay** the credit. Probability: {probability:.2f}")
    else:
        st.error(f"❌ Customer is likely to **default**. Probability: {probability:.2f}")

# ===============================
# Graphs Section
# ===============================
st.header("Feature Analysis & Graphs")

# Example: Debt vs Payment Ratio
st.subheader("Debt Ratio vs Payment Ratio")
fig, ax = plt.subplots()
ax.scatter(input_df['debt_ratio'], input_df['payment_ratio'], color='blue')
ax.set_xlabel("Debt Ratio")
ax.set_ylabel("Payment Ratio")
ax.set_title("Debt Ratio vs Payment Ratio")
st.pyplot(fig)

# Example: Bill Amounts
st.subheader("Bill Amounts Trend")
fig2, ax2 = plt.subplots()
ax2.plot([1,2,3,4,5,6],[input_df['BILL_AMT1'][0], input_df['BILL_AMT2'][0], input_df['BILL_AMT3'][0],
                        input_df['BILL_AMT4'][0], input_df['BILL_AMT5'][0], input_df['BILL_AMT6'][0]], marker='o')
ax2.set_xlabel("Month")
ax2.set_ylabel("Bill Amount")
ax2.set_title("Bill Amounts Over 6 Months")
st.pyplot(fig2)

# Example: Payment Amounts
st.subheader("Payment Amounts Trend")
fig3, ax3 = plt.subplots()
ax3.plot([1,2,3,4,5,6],[input_df['PAY_AMT1'][0], input_df['PAY_AMT2'][0], input_df['PAY_AMT3'][0],
                        input_df['PAY_AMT4'][0], input_df['PAY_AMT5'][0], input_df['PAY_AMT6'][0]], marker='o', color='green')
ax3.set_xlabel("Month")
ax3.set_ylabel("Payment Amount")
ax3.set_title("Payments Over 6 Months")
st.pyplot(fig3)
