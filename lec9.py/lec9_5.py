"""
=========================================================
Lecture 9 - Program 5
Topic : AI Performance Report

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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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
        placement = random.choice(
            ["Placed", "Placed", "Placed", "Not Placed"]
        )
    else:
        placement = random.choice(
            ["Not Placed", "Not Placed", "Placed"]
        )

    student_data["Attendance"].append(attendance)
    student_data["Coding_Score"].append(coding)
    student_data["Study_Hours"].append(study)
    student_data["Communication_Skills"].append(communication)
    student_data["Placement_Status"].append(placement)

# =========================================================
# Create DataFrame
# =========================================================

df = pd.DataFrame(student_data)

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
# Train Model
# =========================================================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    random_state=42
)

model.fit(X_train, y_train)

# =========================================================
# New Student
# =========================================================

student = {
    "Attendance": 85,
    "Coding_Score": 72,
    "Study_Hours": 6,
    "Communication_Skills": 4
}

student_df = pd.DataFrame([student])

# =========================================================
# Prediction
# =========================================================

prediction = model.predict(student_df)

result = encoder.inverse_transform(prediction)

# =========================================================
# Probability
# =========================================================

probability = model.predict_proba(student_df)

placed_index = list(model.classes_).index(1)

placement_probability = probability[0][placed_index]

# =========================================================
# Strength Analysis
# =========================================================

strengths = []

if student["Attendance"] >= 80:
    strengths.append("Excellent Attendance")

if student["Coding_Score"] >= 75:
    strengths.append("Strong Coding Skills")

if student["Study_Hours"] >= 6:
    strengths.append("Consistent Study Habits")

if student["Communication_Skills"] >= 7:
    strengths.append("Good Communication Skills")

# =========================================================
# Weakness Analysis
# =========================================================

weaknesses = []

if student["Attendance"] < 80:
    weaknesses.append("Attendance Needs Improvement")

if student["Coding_Score"] < 75:
    weaknesses.append("Coding Skills Need Improvement")

if student["Study_Hours"] < 6:
    weaknesses.append("Increase Daily Study Hours")

if student["Communication_Skills"] < 7:
    weaknesses.append("Improve Communication Skills")

# =========================================================
# Recommendations
# =========================================================

recommendations = []

if student["Coding_Score"] < 75:
    recommendations.append(
        "Practice DSA and Coding Problems Daily."
    )

if student["Communication_Skills"] < 7:
    recommendations.append(
        "Attend Mock Interviews and Improve Communication."
    )

if student["Attendance"] < 80:
    recommendations.append(
        "Maintain Above 80% Attendance."
    )

if student["Study_Hours"] < 6:
    recommendations.append(
        "Increase Daily Study Time to at least 6 Hours."
    )

# =========================================================
# AI Performance Report
# =========================================================

print("=" * 60)
print("           AI PERFORMANCE REPORT")
print("=" * 60)

print(f"Prediction               : {result[0]}")
print(f"Placement Probability    : {placement_probability * 100:.2f}%")

print("\nStrengths")

if strengths:
    for strength in strengths:
        print(f"✔ {strength}")
else:
    print("No major strengths found.")

print("\nWeaknesses")

if weaknesses:
    for weakness in weaknesses:
        print(f"✘ {weakness}")
else:
    print("No major weaknesses found.")

print("\nRecommendations")

if recommendations:
    for recommendation in recommendations:
        print(f"➜ {recommendation}")
else:
    print("No recommendations required.")

# =========================================================
# Summary DataFrame
# =========================================================

summary = {
    "Prediction": result[0],
    "Placement Probability (%)": round(
        placement_probability * 100,
        2
    ),
    "Total Strengths": len(strengths),
    "Total Weaknesses": len(weaknesses),
    "Recommendations": len(recommendations)
}

summary_df = pd.DataFrame([summary])

print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)

print(summary_df)