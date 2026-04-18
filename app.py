# ==========================================
# STREAMLIT APP - EMPLOYEE PERFORMANCE
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# PAGE SETTINGS
# -------------------------------

st.set_page_config(page_title="Employee Performance Predictor", layout="centered")

st.title("👩‍💼 Employee Performance Predictor")
st.write("Enter employee details to predict performance level")

# -------------------------------
# CREATE DATA
# -------------------------------

@st.cache_data
def create_data():
    np.random.seed(42)
    n = 500

    data = pd.DataFrame({
        "Age": np.random.randint(22, 60, n),
        "Experience": np.random.randint(1, 20, n),
        "Department": np.random.choice(["HR", "IT", "Sales"], n),
        "Salary": np.random.randint(20000, 100000, n),
        "Training_Hours": np.random.randint(10, 100, n),
        "Projects": np.random.randint(1, 10, n)
    })

    performance = []

    for i in range(n):
        score = 0
        
        if data.loc[i, "Experience"] > 10:
            score += 1
        if data.loc[i, "Training_Hours"] > 50:
            score += 1
        if data.loc[i, "Projects"] > 5:
            score += 1
        if data.loc[i, "Salary"] > 50000:
            score += 1

        score += np.random.choice([0, 1])

        if score <= 2:
            performance.append("Low")
        elif score == 3:
            performance.append("Medium")
        else:
            performance.append("High")

    data["Performance"] = performance
    return data

data = create_data()

# -------------------------------
# PREPROCESSING
# -------------------------------

le_dept = LabelEncoder()
le_perf = LabelEncoder()

data["Department"] = le_dept.fit_transform(data["Department"])
data["Performance"] = le_perf.fit_transform(data["Performance"])

X = data.drop("Performance", axis=1)
y = data["Performance"]

# -------------------------------
# TRAIN MODEL
# -------------------------------

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# -------------------------------
# USER INPUT
# -------------------------------

st.subheader("📥 Enter Employee Details")

age = st.slider("Age", 20, 60, 30)
experience = st.slider("Experience (Years)", 1, 20, 5)
department = st.selectbox("Department", ["HR", "IT", "Sales"])
salary = st.slider("Salary", 20000, 100000, 40000)
training = st.slider("Training Hours", 10, 100, 50)
projects = st.slider("Number of Projects", 1, 10, 5)

# Encode department
dept_encoded = le_dept.transform([department])[0]

# -------------------------------
# PREDICTION
# -------------------------------

if st.button("Predict Performance"):
    
    input_data = pd.DataFrame([{
        "Age": age,
        "Experience": experience,
        "Department": dept_encoded,
        "Salary": salary,
        "Training_Hours": training,
        "Projects": projects
    }])

    prediction = model.predict(input_data)
    result = le_perf.inverse_transform(prediction)

    st.success(f"🎯 Predicted Performance: {result[0]}")