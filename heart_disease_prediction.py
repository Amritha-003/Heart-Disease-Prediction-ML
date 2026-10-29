"""
Educational heart disease risk classification example.

This script uses a small synthetic dataset to demonstrate the machine-learning
workflow. It is NOT a clinical model and must not be used for diagnosis or
medical decisions. For meaningful research, use a properly licensed dataset,
follow its documentation, and evaluate models rigorously.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Synthetic demonstration data only (not real patient records).
data = {
    "age": [29, 35, 42, 51, 60, 48, 55, 63, 39, 46, 58, 33, 67, 52, 44, 61, 37, 49, 56, 41],
    "resting_bp": [118, 120, 125, 135, 145, 130, 140, 150, 122, 128, 142, 115, 155, 138, 126, 148, 119, 132, 144, 124],
    "cholesterol": [180, 190, 205, 230, 250, 220, 240, 265, 195, 210, 245, 175, 270, 235, 200, 258, 188, 225, 248, 198],
    "max_heart_rate": [190, 185, 178, 160, 145, 165, 150, 138, 182, 170, 148, 192, 135, 158, 176, 140, 187, 162, 149, 180],
    "target": [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0],
}

df = pd.DataFrame(data)
X = df[["age", "resting_bp", "cholesterol", "max_heart_rate"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=42, max_iter=1000)
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Synthetic demonstration only — not for medical use.")
print(f"Test accuracy: {accuracy_score(y_test, predictions):.2f}")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# Example prediction using made-up values, for code demonstration only.
sample = pd.DataFrame([{
    "age": 50,
    "resting_bp": 130,
    "cholesterol": 220,
    "max_heart_rate": 165,
}])
predicted_class = model.predict(sample)[0]
print("\nExample synthetic prediction class:", int(predicted_class))
print("Class 0/1 is only a label in this toy dataset, not a medical assessment.")
