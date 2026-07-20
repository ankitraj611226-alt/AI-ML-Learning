"""
=========================================================
Lecture 8 - Program 1
Topic : Cross Validation

Author : Ankit Raj
Course : AI & ML Internship
=========================================================
"""

# =========================================================
# Import Libraries
# =========================================================

import random
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# =========================================================
# Generate Dataset
# =========================================================

student_data = {
    "Attendance": [],
    "Coding_Score": [],
    "Study_Hours": [],
    "Communication_Skills": [],
    "Placement_Status": []
}

for _ in range(200):

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

    placement = "Placed" if score >= 80 else "Not Placed"

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
print("First 5 Records")
print("=" * 60)
print(df.head())

# =========================================================
# Label Encoding
# =========================================================

encoder = LabelEncoder()

df["Placement_Status"] = encoder.fit_transform(df["Placement_Status"])

# =========================================================
# Features & Target
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
# Random Forest Model
# =========================================================

model = RandomForestClassifier(random_state=42)

# =========================================================
# Cross Validation
# =========================================================

scores = cross_val_score(
    estimator=model,
    X=X,
    y=y,
    cv=5,
    scoring="accuracy"
)

# =========================================================
# Results
# =========================================================

print("\n" + "=" * 60)
print("Cross Validation Results")
print("=" * 60)

print("Scores           :", scores)
print("Average Accuracy :", round(scores.mean(), 4))
print("Highest Accuracy :", round(scores.max(), 4))
print("Lowest Accuracy  :", round(scores.min(), 4))