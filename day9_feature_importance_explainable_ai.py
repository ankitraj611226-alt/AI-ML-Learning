"""
==============================================================
           Day 09 - Explainable AI & Feature Importance
--------------------------------------------------------------
Author      : Ankit Raj
Language    : Python

Description:
Build an Explainable AI Placement Prediction System using
Random Forest, Feature Importance, Feature Selection,
and Student Performance Analysis.

Topics Covered
--------------
✔ Feature Importance
✔ Feature Selection
✔ Explainable AI (XAI)
✔ Student Performance Analysis
✔ Strength Analysis
✔ Weakness Detection
✔ AI Career Recommendations
✔ Placement Prediction
✔ Performance Report
==============================================================
"""

# ==========================================================
# Import Libraries
# ==========================================================

import random
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ==========================================================
# Set Random Seed
# ==========================================================

random.seed(42)

print("=" * 70)
print("DAY 09 - EXPLAINABLE AI & FEATURE IMPORTANCE")
print("=" * 70)

# ==========================================================
# Generate Student Dataset
# ==========================================================

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
            "Placed",
            "Not Placed",
            "Not Placed"
        ])

    student_data["Attendance"].append(attendance)
    student_data["Coding_Score"].append(coding)
    student_data["Study_Hours"].append(study)
    student_data["Communication_Skills"].append(communication)
    student_data["Placement_Status"].append(placement)

# ==========================================================
# Create DataFrame
# ==========================================================

df = pd.DataFrame(student_data)

print("\nDataset Shape :", df.shape)

print("\nFirst Five Records\n")
print(df.head())

print("\nDataset Information\n")
print(df.info())

print("\nStatistical Summary\n")
print(df.describe())
# ==========================================================
# Label Encoding
# ==========================================================

print("\n" + "=" * 70)
print("LABEL ENCODING")
print("=" * 70)

encoder = LabelEncoder()

df["Placement_Status"] = encoder.fit_transform(
    df["Placement_Status"]
)

print("\nEncoded Dataset\n")
print(df.head())

# ==========================================================
# Features & Target
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
print("Target Vector Shape  :", y.shape)

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
print("Testing Samples  :", len(X_test))

# ==========================================================
# Train Optimized Random Forest Model
# ==========================================================

print("\n" + "=" * 70)
print("TRAINING OPTIMIZED RANDOM FOREST MODEL")
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

# ==========================================================
# Model Accuracy
# ==========================================================

training_accuracy = model.score(
    X_train,
    y_train
)

testing_accuracy = model.score(
    X_test,
    y_test
)

print(f"\nTraining Accuracy : {training_accuracy:.4f}")
print(f"Testing Accuracy  : {testing_accuracy:.4f}")

print("\nModel Training Completed Successfully!")

# ==========================================================
# Feature Summary
# ==========================================================

feature_summary = pd.DataFrame({

    "Feature": X.columns,

    "Description": [

        "Student Attendance Percentage",

        "Coding Assessment Score",

        "Daily Study Hours",

        "Communication Skill Rating"

    ]

})

print("\nFeatures Used By The Model\n")

print(feature_summary)
# ==========================================================
# Feature Importance
# ==========================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

# Calculate Feature Importance

importance = model.feature_importances_

importance_df = pd.DataFrame({

    "Feature": X.columns,

    "Importance": importance

})

# Sort Features

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance Table\n")

print(importance_df)

# ==========================================================
# Visualize Feature Importance
# ==========================================================

plt.figure(figsize=(8,5))

bars = plt.barh(

    importance_df["Feature"],

    importance_df["Importance"]

)

plt.title("Feature Importance Analysis")

plt.xlabel("Importance Score")

plt.ylabel("Features")

plt.gca().invert_yaxis()

# Display Values

for bar in bars:

    width = bar.get_width()

    plt.text(

        width + 0.005,

        bar.get_y() + bar.get_height()/2,

        f"{width:.3f}",

        va="center",

        fontsize=10

    )

plt.tight_layout()

plt.show()

# ==========================================================
# Most & Least Important Features
# ==========================================================

most_important = importance_df.iloc[0]

least_important = importance_df.iloc[-1]

print("\nMost Important Feature")
print("-" * 40)
print(most_important)

print("\nLeast Important Feature")
print("-" * 40)
print(least_important)

# ==========================================================
# Feature Selection
# ==========================================================

print("\n" + "=" * 70)
print("FEATURE SELECTION")
print("=" * 70)

selected_features = importance_df[
    importance_df["Importance"] > 0.15
]

print("\nSelected Features\n")

print(selected_features)

# ==========================================================
# Create New Dataset
# ==========================================================

X_selected = df[
    selected_features["Feature"]
]

print("\nSelected Dataset Shape :", X_selected.shape)

# ==========================================================
# Train-Test Split
# ==========================================================

X_train_selected, X_test_selected, y_train_selected, y_test_selected = train_test_split(

    X_selected,

    y,

    test_size=0.20,

    random_state=42

)

# ==========================================================
# Train Model Using Selected Features
# ==========================================================

selected_model = RandomForestClassifier(

    n_estimators=100,

    max_depth=5,

    min_samples_split=10,

    random_state=42

)

selected_model.fit(
    X_train_selected,
    y_train_selected
)

# ==========================================================
# Accuracy After Feature Selection
# ==========================================================

selected_accuracy = selected_model.score(

    X_test_selected,

    y_test_selected

)

original_accuracy = model.score(
    X_test,
    y_test
)

print(f"\nOriginal Model Accuracy : {original_accuracy:.4f}")

print(f"Selected Feature Accuracy : {selected_accuracy:.4f}")

# ==========================================================
# Accuracy Comparison
# ==========================================================

comparison_df = pd.DataFrame({

    "Model":[

        "All Features",

        "Selected Features"

    ],

    "Accuracy":[

        original_accuracy,

        selected_accuracy

    ]

})

print("\nAccuracy Comparison\n")

print(comparison_df)

# ==========================================================
# Visualization
# ==========================================================

plt.figure(figsize=(7,5))

bars = plt.bar(

    comparison_df["Model"],

    comparison_df["Accuracy"]

)

plt.title("Before vs After Feature Selection")

plt.ylabel("Accuracy")

plt.ylim(0,1)

for bar in bars:

    height = bar.get_height()

    plt.text(

        bar.get_x() + bar.get_width()/2,

        height + 0.01,

        f"{height:.3f}",

        ha="center"

    )

plt.tight_layout()

plt.show()

print("\nFeature Selection Completed Successfully!")
# ==========================================================
# Student Performance Analysis Engine
# ==========================================================

print("\n" + "=" * 70)
print("STUDENT PERFORMANCE ANALYSIS ENGINE")
print("=" * 70)

# ==========================================================
# New Student Record
# ==========================================================

student = {

    "Attendance": 85,

    "Coding_Score": 72,

    "Study_Hours": 6,

    "Communication_Skills": 4

}

student_df = pd.DataFrame([student])

print("\nStudent Profile\n")
print(student_df)

# ==========================================================
# Placement Prediction
# ==========================================================

prediction = model.predict(student_df)

prediction_result = encoder.inverse_transform(prediction)

print("\nPlacement Prediction :", prediction_result[0])

# ==========================================================
# Placement Probability
# ==========================================================

probability = model.predict_proba(student_df)

placed_index = list(model.classes_).index(1)

placement_probability = probability[0][placed_index]

print(
    "Placement Probability :",
    f"{placement_probability * 100:.2f}%"
)

# ==========================================================
# Strength Analysis
# ==========================================================

strengths = []

if student["Attendance"] >= 80:
    strengths.append("Excellent Attendance")

if student["Coding_Score"] >= 75:
    strengths.append("Strong Coding Skills")

if student["Study_Hours"] >= 6:
    strengths.append("Consistent Study Habits")

if student["Communication_Skills"] >= 7:
    strengths.append("Good Communication Skills")

# ==========================================================
# Weakness Analysis
# ==========================================================

weaknesses = []

if student["Attendance"] < 80:
    weaknesses.append("Attendance Needs Improvement")

if student["Coding_Score"] < 75:
    weaknesses.append("Coding Skills Need Improvement")

if student["Study_Hours"] < 6:
    weaknesses.append("Increase Daily Study Hours")

if student["Communication_Skills"] < 7:
    weaknesses.append("Improve Communication Skills")

# ==========================================================
# AI Career Recommendations
# ==========================================================

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

# ==========================================================
# Display AI Analysis
# ==========================================================

print("\n" + "=" * 70)
print("AI PERFORMANCE ANALYSIS")
print("=" * 70)

print("\nStrengths")

if strengths:

    for index, strength in enumerate(strengths, start=1):
        print(f"{index}. {strength}")

else:
    print("No strengths identified.")

print("\nWeaknesses")

if weaknesses:

    for index, weakness in enumerate(weaknesses, start=1):
        print(f"{index}. {weakness}")

else:
    print("No weaknesses identified.")

print("\nAI Career Recommendations")

if recommendations:

    for index, recommendation in enumerate(
        recommendations,
        start=1
    ):
        print(f"{index}. {recommendation}")

else:
    print("No recommendations required.")

# ==========================================================
# Student Analysis Summary
# ==========================================================

analysis_summary = pd.DataFrame({

    "Prediction": [prediction_result[0]],

    "Placement Probability (%)": [
        round(placement_probability * 100, 2)
    ],

    "Total Strengths": [
        len(strengths)
    ],

    "Total Weaknesses": [
        len(weaknesses)
    ],

    "Recommendations": [
        len(recommendations)
    ]

})

print("\nAnalysis Summary\n")
print(analysis_summary)

print("\nStudent Performance Analysis Completed Successfully!")# ==========================================================
# AI Performance Report
# ==========================================================

print("\n" + "=" * 70)
print("AI PERFORMANCE REPORT")
print("=" * 70)

print(f"\nPrediction               : {prediction_result[0]}")

print(
    f"Placement Probability    : "
    f"{placement_probability * 100:.2f}%"
)

print("\nStrengths")
print("-" * 40)

if strengths:
    for strength in strengths:
        print(f"✔ {strength}")
else:
    print("No strengths identified.")

print("\nWeaknesses")
print("-" * 40)

if weaknesses:
    for weakness in weaknesses:
        print(f"✖ {weakness}")
else:
    print("No weaknesses identified.")

print("\nRecommendations")
print("-" * 40)

if recommendations:
    for recommendation in recommendations:
        print(f"➜ {recommendation}")
else:
    print("No recommendations required.")

# ==========================================================
# Explainable AI Workflow
# ==========================================================

print("\n" + "=" * 70)
print("EXPLAINABLE AI WORKFLOW")
print("=" * 70)

workflow = [
    "Student Data",
    "Feature Processing",
    "Placement Prediction",
    "Placement Probability",
    "Feature Importance",
    "Strength Analysis",
    "Weakness Analysis",
    "AI Career Recommendations"
]

for step in workflow:
    print(f"↓ {step}")

# ==========================================================
# Final Project Summary
# ==========================================================

summary = {

    "Prediction": prediction_result[0],

    "Placement Probability (%)": round(
        placement_probability * 100,
        2
    ),

    "Total Strengths": len(strengths),

    "Total Weaknesses": len(weaknesses),

    "Recommendations": len(recommendations)

}

summary_df = pd.DataFrame([summary])

print("\nProject Summary\n")
print(summary_df)

# ==========================================================
# Save Report
# ==========================================================

summary_df.to_csv(
    "ai_performance_report.csv",
    index=False
)

print("\nAI Performance Report saved successfully.")

# ==========================================================
# Project Features
# ==========================================================

print("\n" + "=" * 70)
print("PROJECT FEATURES")
print("=" * 70)

features = [

    "Placement Prediction",

    "Placement Probability",

    "Feature Importance",

    "Feature Selection",

    "Explainable AI",

    "Student Performance Analysis",

    "Strength Detection",

    "Weakness Detection",

    "AI Career Recommendations",

    "Performance Report Generation"

]

for feature in features:
    print(f"✔ {feature}")

# ==========================================================
# Completion Message
# ==========================================================

print("\n" + "=" * 70)
print("DAY 09 PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)

print("""
Congratulations!

You have successfully built an Explainable AI
Placement Prediction System capable of:

• Predicting Placement Status
• Calculating Placement Probability
• Identifying Important Features
• Selecting Useful Features
• Explaining AI Decisions
• Detecting Student Strengths
• Identifying Weak Areas
• Generating Personalized Career Advice
• Exporting an AI Performance Report

This is similar to the type of Explainable AI
systems used in modern AI and Machine Learning
applications.
""")