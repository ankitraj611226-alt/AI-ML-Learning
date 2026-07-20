"""
=========================================================
Lecture 10 - Program 3
Topic : AI Prediction Pipeline

Author : Ankit Raj
Course : AI & ML Internship
=========================================================
"""

# =========================================================
# Import Libraries
# =========================================================

import pickle
import pandas as pd

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
# Prediction Function
# =========================================================

def predict_student(
    attendance,
    coding_score,
    study_hours,
    communication_skills
):
    """
    Predicts placement status of a student.
    """

    # =============================================
    # Create Input Data
    # =============================================

    student = pd.DataFrame([{
        "Attendance": attendance,
        "Coding_Score": coding_score,
        "Study_Hours": study_hours,
        "Communication_Skills": communication_skills
    }])

    # =============================================
    # Predict Class
    # =============================================

    prediction = loaded_model.predict(student)

    # =============================================
    # Decode Prediction
    # =============================================

    result = loaded_encoder.inverse_transform(prediction)

    # =============================================
    # Prediction Probability
    # =============================================

    probability = loaded_model.predict_proba(student)

    placed_index = list(
        loaded_model.classes_
    ).index(1)

    placement_probability = probability[0][placed_index]

    # =============================================
    # Display Result
    # =============================================

    print("\n" + "=" * 60)
    print("STUDENT PLACEMENT REPORT")
    print("=" * 60)

    print(f"Attendance           : {attendance}%")
    print(f"Coding Score         : {coding_score}")
    print(f"Study Hours          : {study_hours}")
    print(f"Communication Skills : {communication_skills}")

    print("-" * 60)

    print("Prediction           :", result[0])

    print(
        "Placement Probability:",
        round(
            placement_probability * 100,
            2
        ),
        "%"
    )

    print("=" * 60)

# =========================================================
# Test Prediction
# =========================================================

predict_student(
    attendance=85,
    coding_score=78,
    study_hours=7,
    communication_skills=8
)