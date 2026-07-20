"""
=========================================================
Lecture 8 - Program 5
Topic : Predict Probability

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

    attendance = random.randint(40,100)
    coding = random.randint(20,100)
    study = random.randint(1,10)
    communication = random.randint(1,10)

    score = (
        coding*0.5 +
        attendance*0.2 +
        study*5 +
        communication*5
    )

    placement = "Placed" if score >= 80 else "Not Placed"

    student_data["Attendance"].append(attendance)
    student_data["Coding_Score"].append(coding)
    student_data["Study_Hours"].append(study)
    student_data["Communication_Skills"].append(communication)
    student_data["Placement_Status"].append(placement)

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
# Train Random Forest Model
# =========================================================

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# =========================================================
# New Student
# =========================================================

new_student = pd.DataFrame({

    "Attendance":[85],
    "Coding_Score":[90],
    "Study_Hours":[7],
    "Communication_Skills":[8]

})

# =========================================================
# Prediction
# =========================================================

prediction = model.predict(new_student)

probability = model.predict_proba(new_student)

# =========================================================
# Results
# =========================================================

print("="*60)
print("Prediction Result")
print("="*60)

print("Prediction :", prediction)

print("\nProbability :", probability)

print("\nNot Placed Probability :",
      round(probability[0][0]*100,2),
      "%")

print("Placed Probability     :",
      round(probability[0][1]*100,2),
      "%")