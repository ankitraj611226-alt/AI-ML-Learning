"""
=========================================================
Lecture 10 - Program 1
Topic : Model Serialization using Pickle

Author : Ankit Raj
Course : AI & ML Internship
=========================================================
"""

# =========================================================
# Import Libraries
# =========================================================

import random
import pickle
import os
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================================================
# Generate Student Dataset
# =========================================================

student_data = {
    "Attendance": [],
    "Coding_Score": [],
    "Study_Hours": [],
    "Communication_Skills": [],
    "Placement_Status": []
}

for _ in range(500):

    attendance = random.randint(40, 100)
    coding = random.randint(20, 100)
    study = random.randint(1, 10)
    communication = random.randint(1, 10)

    score = (
        coding * 0.5 +
        attendance * 0.2 +
        study * 5 +
        communication * 5
    )

    if score >= 80:
        placement = random.choice([
            "Placed",
            "Placed",
            "Placed",
            "Not Placed"
        ])
    else:
        placement = random.choice([
            "Not Placed",
            "Not Placed",
            "Placed"
        ])

    student_data["Attendance"].append(attendance)
    student_data["Coding_Score"].append(coding)
    student_data["Study_Hours"].append(study)
    student_data["Communication_Skills"].append(communication)
    student_data["Placement_Status"].append(placement)

# =========================================================
# Create DataFrame
# =========================================================

df = pd.DataFrame(student_data)

print("=" * 60)
print("Student Dataset")
print("=" * 60)

print(df.head())

# =========================================================
# Label Encoding
# =========================================================

encoder = LabelEncoder()

df["Placement_Status"] = encoder.fit_transform(
    df["Placement_Status"]
)

# =========================================================
# Features and Target
# =========================================================

X = df[
    [
        "Attendance",
        "Coding_Score",
        "Study_Hours",
        "Communication_Skills"
    ]
]

y = df["Placement_Status"]

# =========================================================
# Train Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =========================================================
# Train Random Forest Model
# =========================================================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully.")

# =========================================================
# Evaluate Model
# =========================================================

prediction = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    prediction
)

print(
    "Model Accuracy :",
    round(accuracy * 100, 2),
    "%"
)

# =========================================================
# Save Trained Model
# =========================================================

with open("placement_model.pkl", "wb") as file:

    pickle.dump(model, file)

print("\nModel Saved Successfully.")

# =========================================================
# Save Label Encoder
# =========================================================

with open("label_encoder.pkl", "wb") as file:

    pickle.dump(encoder, file)

print("Label Encoder Saved Successfully.")

# =========================================================
# Verify Saved Files
# =========================================================

print("\nChecking Saved Files...")

print(
    "Model Exists :",
    os.path.exists("placement_model.pkl")
)

print(
    "Encoder Exists :",
    os.path.exists("label_encoder.pkl")
)

# =========================================================
# Program Completed
# =========================================================

print("\n" + "=" * 60)
print("Model Serialization Completed Successfully")
print("=" * 60)