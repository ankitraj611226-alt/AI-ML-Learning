"""
=========================================================
Lecture 10 - Program 2
Topic : Load Saved Model and Make Predictions

Author : Ankit Raj
Course : AI & ML Internship
=========================================================
"""

# =========================================================
# Import Libraries
# =========================================================

import random
import pickle
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
# Save Model
# =========================================================

with open("placement_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("label_encoder.pkl", "wb") as file:
    pickle.dump(encoder, file)

print("=" * 60)
print("Model and Encoder Saved Successfully")
print("=" * 60)

# =========================================================
# Load Saved Model
# =========================================================

with open("placement_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

print("Model Loaded Successfully.")

# =========================================================
# Load Saved Label Encoder
# =========================================================

with open("label_encoder.pkl", "rb") as file:
    loaded_encoder = pickle.load(file)

print("Label Encoder Loaded Successfully.")

# =========================================================
# Verify Loaded Model Accuracy
# =========================================================

loaded_prediction = loaded_model.predict(X_test)

loaded_accuracy = accuracy_score(
    y_test,
    loaded_prediction
)

print("\nLoaded Model Accuracy :",
      round(loaded_accuracy * 100, 2), "%")

# =========================================================
# Predict New Student
# =========================================================

new_student = pd.DataFrame([{
    "Attendance": 85,
    "Coding_Score": 78,
    "Study_Hours": 7,
    "Communication_Skills": 8
}])

prediction = loaded_model.predict(new_student)

result = loaded_encoder.inverse_transform(prediction)

print("\nPrediction :", result[0])

# =========================================================
# Predict Placement Probability
# =========================================================

probability = loaded_model.predict_proba(new_student)

placed_index = list(
    loaded_model.classes_
).index(1)

placement_probability = probability[0][placed_index]

print(
    "Placement Probability :",
    round(placement_probability * 100, 2),
    "%"
)

# =========================================================
# Program Completed
# =========================================================

print("\n" + "=" * 60)
print("Model Loading Completed Successfully")
print("=" * 60)