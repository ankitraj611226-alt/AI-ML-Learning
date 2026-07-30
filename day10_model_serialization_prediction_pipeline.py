"""
==============================================================
AI/ML Internship - Day 10
Project: Model Serialization & AI Prediction Pipeline
Author: Ankit Raj
==============================================================

Project Objectives:
1. Generate a realistic student placement dataset
2. Preprocess the data
3. Train a Random Forest Classifier
4. Evaluate model performance
5. Prepare the model for serialization

Technologies Used:
- Python
- Pandas
- Scikit-learn
- Random Forest
- Pickle
"""

# ==========================================================
# Import Required Libraries
# ==========================================================

import random
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================================
# Generate Student Dataset
# ==========================================================

print("=" * 70)
print("GENERATING STUDENT DATASET")
print("=" * 70)

random.seed(42)

student_data = {

    "Attendance": [],
    "Coding_Score": [],
    "Study_Hours": [],
    "Communication_Skills": [],
    "Placement_Status": []

}

for _ in range(500):

    attendance = random.randint(40, 100)
    coding_score = random.randint(20, 100)
    study_hours = random.randint(1, 10)
    communication_skills = random.randint(1, 10)

    score = (

        coding_score * 0.50 +

        attendance * 0.20 +

        study_hours * 5 +

        communication_skills * 5

    )

    if score >= 80:

        placement = random.choice(

            [

                "Placed",

                "Placed",

                "Placed",

                "Not Placed"

            ]

        )

    else:

        placement = random.choice(

            [

                "Not Placed",

                "Not Placed",

                "Placed"

            ]

        )

    student_data["Attendance"].append(attendance)
    student_data["Coding_Score"].append(coding_score)
    student_data["Study_Hours"].append(study_hours)
    student_data["Communication_Skills"].append(communication_skills)
    student_data["Placement_Status"].append(placement)

# ==========================================================
# Create DataFrame
# ==========================================================

df = pd.DataFrame(student_data)

print("\nDataset Created Successfully!")

print("\nFirst Five Records\n")

print(df.head())

# ==========================================================
# Dataset Information
# ==========================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(df.info())

print("\nDataset Shape :", df.shape)

print("\nColumn Names")

print(df.columns.tolist())

# ==========================================================
# Statistical Summary
# ==========================================================

print("\n" + "=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)

print(df.describe())

# ==========================================================
# Encode Target Variable
# ==========================================================

print("\n" + "=" * 70)
print("LABEL ENCODING")
print("=" * 70)

encoder = LabelEncoder()

df["Placement_Status"] = encoder.fit_transform(

    df["Placement_Status"]

)

print(df.head())

print("\nEncoding Completed Successfully!")

# ==========================================================
# Feature Matrix & Target Variable
# ==========================================================

X = df[

    [

        "Attendance",

        "Coding_Score",

        "Study_Hours",

        "Communication_Skills"

    ]

]

y = df["Placement_Status"]

print("\nFeature Matrix Shape :", X.shape)

print("Target Variable Shape :", y.shape)

# ==========================================================
# Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)

print("\nTraining Samples :", len(X_train))

print("Testing Samples :", len(X_test))

# ==========================================================
# Train Random Forest Model
# ==========================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST MODEL")
print("=" * 70)

model = RandomForestClassifier(

    n_estimators=100,

    max_depth=5,

    min_samples_split=10,

    random_state=42

)

model.fit(

    X_train,

    y_train

)

print("Model Trained Successfully!")

# ==========================================================
# Model Evaluation
# ==========================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(

    y_test,

    predictions

)

print("\nModel Accuracy :", round(accuracy, 4))

print("\nDay 10 - Part 1 Completed Successfully!")
# ==========================================================
# Why Model Serialization?
# ==========================================================

print("\n" + "=" * 70)
print("MODEL SERIALIZATION")
print("=" * 70)

print("""
Training a Machine Learning model every time a prediction
is required is inefficient.

Instead, AI engineers train the model once, save it,
and reuse it whenever predictions are needed.

This process is called Model Serialization.
""")

# ==========================================================
# Import Pickle
# ==========================================================

import pickle
import os

# ==========================================================
# Save Trained Model
# ==========================================================

print("\nSaving Random Forest Model...")

with open("placement_model.pkl", "wb") as file:

    pickle.dump(model, file)

print("✔ Model Saved Successfully!")

# ==========================================================
# Save Label Encoder
# ==========================================================

print("\nSaving Label Encoder...")

with open("label_encoder.pkl", "wb") as file:

    pickle.dump(encoder, file)

print("✔ Label Encoder Saved Successfully!")

# ==========================================================
# Verify Saved Files
# ==========================================================

print("\n" + "=" * 70)
print("VERIFY SAVED FILES")
print("=" * 70)

model_exists = os.path.exists("placement_model.pkl")
encoder_exists = os.path.exists("label_encoder.pkl")

print("Model File Exists   :", model_exists)
print("Encoder File Exists :", encoder_exists)

# ==========================================================
# File Information
# ==========================================================

if model_exists:

    print(
        "Model File Size :",
        round(
            os.path.getsize("placement_model.pkl") / 1024,
            2
        ),
        "KB"
    )

if encoder_exists:

    print(
        "Encoder File Size :",
        round(
            os.path.getsize("label_encoder.pkl") / 1024,
            2
        ),
        "KB"
    )

# ==========================================================
# Load Saved Model
# ==========================================================

print("\n" + "=" * 70)
print("LOADING SAVED MODEL")
print("=" * 70)

with open("placement_model.pkl", "rb") as file:

    loaded_model = pickle.load(file)

print("✔ Model Loaded Successfully!")

# ==========================================================
# Load Saved Encoder
# ==========================================================

with open("label_encoder.pkl", "rb") as file:

    loaded_encoder = pickle.load(file)

print("✔ Label Encoder Loaded Successfully!")

# ==========================================================
# Evaluate Loaded Model
# ==========================================================

loaded_predictions = loaded_model.predict(X_test)

loaded_accuracy = accuracy_score(

    y_test,

    loaded_predictions

)

print("\nLoaded Model Accuracy :", round(loaded_accuracy, 4))

# ==========================================================
# Accuracy Comparison
# ==========================================================

print("\n" + "=" * 70)
print("ACCURACY COMPARISON")
print("=" * 70)

comparison_df = pd.DataFrame({

    "Model": [

        "Original Model",

        "Loaded Model"

    ],

    "Accuracy": [

        round(accuracy, 4),

        round(loaded_accuracy, 4)

    ]

})

print(comparison_df)

# ==========================================================
# Accuracy Visualization
# ==========================================================

plt.figure(figsize=(6,5))

bars = plt.bar(

    comparison_df["Model"],

    comparison_df["Accuracy"]

)

plt.title("Original vs Loaded Model Accuracy")

plt.ylabel("Accuracy")

plt.ylim(0, 1)

for bar in bars:

    height = bar.get_height()

    plt.text(

        bar.get_x() + bar.get_width() / 2,

        height + 0.01,

        f"{height:.4f}",

        ha="center"

    )

plt.tight_layout()

plt.show()

# ==========================================================
# Conclusion
# ==========================================================

if round(accuracy, 4) == round(loaded_accuracy, 4):

    print("\n✔ Serialization Successful!")

    print("The loaded model performs exactly like the original model.")

else:

    print("\n⚠ Accuracy Mismatch Detected!")

print("\nDay 10 - Part 2 Completed Successfully!")
# ==========================================================
# AI Prediction Pipeline
# ==========================================================

print("\n" + "=" * 70)
print("AI PREDICTION PIPELINE")
print("=" * 70)

print("""
Industry Workflow

Student Data
      ↓
Load Saved Model
      ↓
Placement Prediction
      ↓
Probability Prediction
      ↓
Final Result
""")

# ==========================================================
# Predict a New Student
# ==========================================================

print("\n" + "=" * 70)
print("PREDICTING NEW STUDENT")
print("=" * 70)

new_student = pd.DataFrame({

    "Attendance":[85],
    "Coding_Score":[78],
    "Study_Hours":[7],
    "Communication_Skills":[8]

})

print("\nStudent Details\n")

print(new_student)

# ==========================================================
# Placement Prediction
# ==========================================================

prediction = loaded_model.predict(new_student)

result = loaded_encoder.inverse_transform(prediction)

print("\nPrediction :", result[0])

# ==========================================================
# Prediction Probability
# ==========================================================

probability = loaded_model.predict_proba(new_student)

placed_index = list(loaded_model.classes_).index(1)

placement_probability = probability[0][placed_index]

print(

    "Placement Probability :",

    f"{placement_probability*100:.2f}%"

)

# ==========================================================
# Build Reusable Prediction Function
# ==========================================================

print("\n" + "=" * 70)
print("CREATING REUSABLE PREDICTION FUNCTION")
print("=" * 70)


def predict_student(

        attendance,

        coding_score,

        study_hours,

        communication_skills

):

    student = pd.DataFrame({

        "Attendance":[attendance],

        "Coding_Score":[coding_score],

        "Study_Hours":[study_hours],

        "Communication_Skills":[communication_skills]

    })

    prediction = loaded_model.predict(student)

    result = loaded_encoder.inverse_transform(prediction)

    probability = loaded_model.predict_proba(student)

    placed_index = list(

        loaded_model.classes_

    ).index(1)

    placement_probability = probability[0][placed_index]

    print("\n" + "-" * 60)

    print("Student Profile")

    print("-" * 60)

    print(student)

    print("\nPrediction :", result[0])

    print(

        "Placement Probability :",

        f"{placement_probability*100:.2f}%"

    )

# ==========================================================
# Test Prediction Function
# ==========================================================

predict_student(

    attendance=90,

    coding_score=85,

    study_hours=8,

    communication_skills=9

)

# ==========================================================
# Batch Prediction
# ==========================================================

print("\n" + "=" * 70)
print("BATCH PREDICTION")
print("=" * 70)

students = pd.DataFrame({

    "Attendance":[90,75,60,95,82],

    "Coding_Score":[85,65,45,92,70],

    "Study_Hours":[8,5,3,9,6],

    "Communication_Skills":[9,6,4,10,7]

})

predictions = loaded_model.predict(students)

results = loaded_encoder.inverse_transform(predictions)

probabilities = loaded_model.predict_proba(students)

placed_index = list(

    loaded_model.classes_

).index(1)

# ==========================================================
# Generate AI Report
# ==========================================================

report = students.copy()

report["Prediction"] = results

report["Placement Probability (%)"] = [

    round(

        probability[placed_index]*100,

        2

    )

    for probability in probabilities

]

print("\nAI Prediction Report\n")

print(report)

# ==========================================================
# Save Report
# ==========================================================

report.to_csv(

    "batch_prediction_report.csv",

    index=False

)

print(

    "\nBatch Prediction Report Saved Successfully!"

)

# ==========================================================
# Report Summary
# ==========================================================

print("\n" + "=" * 70)
print("REPORT SUMMARY")
print("=" * 70)

placed = (

    report["Prediction"] == "Placed"

).sum()

not_placed = (

    report["Prediction"] == "Not Placed"

).sum()

print("Total Students :", len(report))

print("Placed :", placed)

print("Not Placed :", not_placed)

average_probability = report[

    "Placement Probability (%)"

].mean()

print(

    "Average Placement Probability :",

    f"{average_probability:.2f}%"

)

print("\nDay 10 - Part 3 Completed Successfully!")
# ==========================================================
# Production Ready Prediction Pipeline
# ==========================================================

print("\n" + "=" * 70)
print("PRODUCTION READY PREDICTION PIPELINE")
print("=" * 70)

# ==========================================================
# Input Validation Function
# ==========================================================

def validate_student(
        attendance,
        coding_score,
        study_hours,
        communication_skills
):

    if not (0 <= attendance <= 100):
        return False, "Attendance must be between 0 and 100."

    if not (0 <= coding_score <= 100):
        return False, "Coding Score must be between 0 and 100."

    if not (0 <= study_hours <= 24):
        return False, "Study Hours must be between 0 and 24."

    if not (0 <= communication_skills <= 10):
        return False, "Communication Skills must be between 0 and 10."

    return True, "Valid Input"

# ==========================================================
# Safe Prediction Function
# ==========================================================

def safe_predict(
        attendance,
        coding_score,
        study_hours,
        communication_skills
):

    print("\n" + "=" * 70)
    print("SAFE PREDICTION")
    print("=" * 70)

    try:

        attendance = float(attendance)
        coding_score = float(coding_score)
        study_hours = float(study_hours)
        communication_skills = float(communication_skills)

        valid, message = validate_student(
            attendance,
            coding_score,
            study_hours,
            communication_skills
        )

        if not valid:

            print("\nValidation Failed")
            print(message)
            return

        student = pd.DataFrame({

            "Attendance":[attendance],
            "Coding_Score":[coding_score],
            "Study_Hours":[study_hours],
            "Communication_Skills":[communication_skills]

        })

        prediction = loaded_model.predict(student)

        result = loaded_encoder.inverse_transform(prediction)

        probability = loaded_model.predict_proba(student)

        placed_index = list(
            loaded_model.classes_
        ).index(1)

        placement_probability = probability[0][placed_index]

        print("\nPrediction Result")
        print("-" * 40)

        print("Prediction :", result[0])

        print(
            "Placement Probability :",
            f"{placement_probability * 100:.2f}%"
        )

    except Exception as error:

        print("\nUnexpected Error")
        print(error)

# ==========================================================
# Valid Input Example
# ==========================================================

print("\nTesting Valid Student")

safe_predict(

    90,

    88,

    8,

    9

)

# ==========================================================
# Invalid Attendance
# ==========================================================

print("\nTesting Invalid Attendance")

safe_predict(

    120,

    80,

    7,

    8

)

# ==========================================================
# Invalid Coding Score
# ==========================================================

print("\nTesting Invalid Coding Score")

safe_predict(

    90,

    140,

    7,

    8

)

# ==========================================================
# Invalid Data Type
# ==========================================================

print("\nTesting Invalid Data Type")

safe_predict(

    "ABC",

    85,

    7,

    8

)

# ==========================================================
# Production ML Workflow
# ==========================================================

print("\n" + "=" * 70)
print("PRODUCTION MACHINE LEARNING WORKFLOW")
print("=" * 70)

workflow = [

    "Receive User Input",

    "Validate Input",

    "Load Saved Model",

    "Predict Placement",

    "Calculate Probability",

    "Generate AI Report",

    "Return Prediction"

]

for step_number, step in enumerate(workflow, start=1):

    print(f"{step_number}. {step}")

# ==========================================================
# Industry Best Practices
# ==========================================================

print("\n" + "=" * 70)
print("INDUSTRY BEST PRACTICES")
print("=" * 70)

best_practices = [

    "Train the model only once.",

    "Save models using Pickle or Joblib.",

    "Validate every user input.",

    "Handle runtime errors using try-except.",

    "Build reusable prediction functions.",

    "Never retrain for every prediction.",

    "Use the same preprocessing during prediction.",

    "Keep prediction pipelines modular."

]

for practice in best_practices:

    print(f"✔ {practice}")

print("\nDay 10 - Part 4 Completed Successfully!")
# ==========================================================
# Final Project Summary
# ==========================================================

print("\n" + "=" * 70)
print("DAY 10 PROJECT SUMMARY")
print("=" * 70)

project_summary = {

    "Dataset Size": len(df),

    "Training Samples": len(X_train),

    "Testing Samples": len(X_test),

    "Original Accuracy": round(accuracy, 4),

    "Loaded Model Accuracy": round(loaded_accuracy, 4),

    "Model Serialized": model_exists,

    "Encoder Serialized": encoder_exists,

    "Batch Prediction Report": "Generated"

}

summary_df = pd.DataFrame([project_summary])

print(summary_df)

# ==========================================================
# Save Project Summary
# ==========================================================

summary_df.to_csv(

    "day10_project_summary.csv",

    index=False

)

print("\nProject Summary Saved Successfully!")

# ==========================================================
# Current Project Architecture
# ==========================================================

print("\n" + "=" * 70)
print("CURRENT PROJECT ARCHITECTURE")
print("=" * 70)

architecture = [

    "Student Dataset",

    "Data Preprocessing",

    "Feature Engineering",

    "Model Training",

    "Model Evaluation",

    "Model Serialization",

    "Load Saved Model",

    "Prediction Pipeline",

    "Batch Prediction",

    "Error Handling",

    "Input Validation",

    "AI Report Generation"

]

for step_number, step in enumerate(architecture, start=1):

    print(f"{step_number}. {step}")

# ==========================================================
# Skills Learned
# ==========================================================

print("\n" + "=" * 70)
print("SKILLS LEARNED")
print("=" * 70)

skills = [

    "Random Forest Classifier",

    "Model Serialization using Pickle",

    "Loading Saved Models",

    "Prediction Pipeline",

    "Batch Prediction",

    "Input Validation",

    "Error Handling",

    "Production Ready ML Workflow",

    "Reusable Python Functions",

    "Industry Machine Learning Practices"

]

for skill in skills:

    print(f"✔ {skill}")

# ==========================================================
# Generated Files
# ==========================================================

print("\n" + "=" * 70)
print("GENERATED PROJECT FILES")
print("=" * 70)

generated_files = [

    "placement_model.pkl",

    "label_encoder.pkl",

    "batch_prediction_report.csv",

    "day10_project_summary.csv"

]

for file in generated_files:

    print(f"📄 {file}")

# ==========================================================
# Project Statistics
# ==========================================================

print("\n" + "=" * 70)
print("PROJECT STATISTICS")
print("=" * 70)

print(f"Dataset Records          : {len(df)}")
print(f"Features Used            : {len(X.columns)}")
print(f"Target Variable          : Placement_Status")
print(f"Machine Learning Model   : Random Forest")
print(f"Prediction Pipeline      : Implemented")
print(f"Model Serialization      : Completed")
print(f"Batch Prediction         : Completed")
print(f"Input Validation         : Completed")
print(f"Error Handling           : Completed")

# ==========================================================
# Final Achievement
# ==========================================================

print("\n" + "=" * 70)
print("DAY 10 COMPLETED SUCCESSFULLY")
print("=" * 70)

