# ==========================================
# EMPLOYEE PERFORMANCE PREDICTOR (FINAL FLOW)
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# STEP 1: CREATE SYNTHETIC DATA
# -------------------------------

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

# -------------------------------
# STEP 2: REALISTIC PERFORMANCE
# -------------------------------

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

    score += np.random.choice([0, 1])  # randomness

    if score <= 2:
        performance.append("Low")
    elif score == 3:
        performance.append("Medium")
    else:
        performance.append("High")

data["Performance"] = performance

# Create folders
os.makedirs("data", exist_ok=True)
os.makedirs("images", exist_ok=True)

# Save dataset
data.to_csv("data/employee_data.csv", index=False)

print("\nDataset Preview:\n")
print(data.head())

# -------------------------------
# STEP 3: PREPROCESSING
# -------------------------------

le = LabelEncoder()
data["Department"] = le.fit_transform(data["Department"])
data["Performance"] = le.fit_transform(data["Performance"])

X = data.drop("Performance", axis=1)
y = data["Performance"]

# -------------------------------
# STEP 4: TRAIN TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# STEP 5: MODEL TRAINING
# -------------------------------

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# STEP 6: PREDICTION
# -------------------------------

y_pred = model.predict(X_test)

# -------------------------------
# STEP 7: EVALUATION
# -------------------------------

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# STEP 8: VISUALIZATION
# -------------------------------

# Performance Distribution (No warning fix applied)
sns.countplot(x=data["Performance"], hue=data["Performance"], palette="Set2", legend=False)
plt.title("Employee Performance Distribution")
plt.savefig("images/performance_distribution.png")
plt.show()

# Feature Importance
importance = model.feature_importances_

plt.figure(figsize=(8, 5))
plt.bar(X.columns, importance)
plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.savefig("images/feature_importance.png")
plt.show()

# -------------------------------
# STEP 9: SAMPLE PREDICTION
# -------------------------------

sample = pd.DataFrame([{
    "Age": 30,
    "Experience": 5,
    "Department": 1,  # HR=0, IT=1, Sales=2
    "Salary": 40000,
    "Training_Hours": 60,
    "Projects": 6
}])

prediction = model.predict(sample)

# Convert numeric prediction to label
label_map = {0: "Low", 1: "Medium", 2: "High"}

print("\nSample Employee Prediction:", label_map[prediction[0]])

print("\nProject Completed Successfully!")
