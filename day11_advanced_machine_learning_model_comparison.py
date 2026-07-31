# ==========================================================
# PROJECT : Student Placement Prediction & Performance Analysis
# FILE : day11_advanced_machine_learning_model_comparison.py
# PART 1 : Import Libraries & Generate Advanced Dataset
# ==========================================================

# ----------------------------------------------------------
# Import Libraries
# ----------------------------------------------------------

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Set Random Seed
# ----------------------------------------------------------

random.seed(42)
np.random.seed(42)

# ----------------------------------------------------------
# Create Empty Dataset
# ----------------------------------------------------------

student_data = {
    'Attendance': [],
    'Coding_Score': [],
    'DSA_Score': [],
    'Aptitude_Score': [],
    'CGPA': [],
    'Study_Hours': [],
    'Communication_Skills': [],
    'Projects': [],
    'Internship': [],
    'Mock_Interview': [],
    'Certification_Count': [],
    'Hackathon_Participation': [],
    'Resume_Score': [],
    'Backlogs': [],
    'Python_Score': [],
    'SQL_Score': [],
    'Problem_Solving': [],
    'GitHub_Activity': [],
    'Placement_Status': []
}

# ----------------------------------------------------------
# Generate 5000 Student Records
# ----------------------------------------------------------

for i in range(5000):

    attendance = random.randint(40, 100)
    coding = random.randint(20, 100)
    dsa = random.randint(20, 100)
    aptitude = random.randint(20, 100)
    cgpa = round(random.uniform(5.0, 10.0), 2)
    study = random.randint(1, 10)
    communication = random.randint(1, 10)
    projects = random.randint(0, 6)
    internship = random.randint(0, 1)
    mock = random.randint(1, 10)
    certifications = random.randint(0, 8)
    hackathon = random.randint(0, 5)
    resume = random.randint(40, 100)
    python_score = random.randint(20, 100)
    sql_score = random.randint(20, 100)
    problem_solving = random.randint(1, 10)
    github = random.randint(0, 100)
    backlogs = random.randint(0, 5)

    score = (
        coding * 0.22 +
        dsa * 0.18 +
        aptitude * 0.15 +
        attendance * 0.08 +
        resume * 0.12 +
        communication * 3 +
        study * 2 +
        mock * 3 +
        cgpa * 5 +
        projects * 4 +
        certifications * 2 +
        hackathon * 3 +
        python_score * 0.15 +
        sql_score * 0.10 +
        problem_solving * 4 +
        github * 0.05
    )

    if internship == 1:
        score += 12

    if projects >= 4:
        score += 10

    if coding > 85 and dsa > 80:
        score += 15

    if communication >= 8:
        score += 8

    if cgpa >= 8.5:
        score += 10

    score -= backlogs * 15

    if coding > 90 and projects >= 5:
        score += 20

    if internship == 1 and communication >= 8:
        score += 15

    if cgpa < 6.5 and backlogs >= 3:
        score -= 30

    if certifications >= 5 and hackathon >= 2:
        score += 12

    score += random.randint(-12, 12)

    if score >= 155:
        placement = random.choice([
            'Placed',
            'Placed',
            'Placed',
            'Placed',
            'Not Placed'
        ])

    elif score >= 135:
        placement = random.choice([
            'Placed',
            'Placed',
            'Placed',
            'Not Placed'
        ])

    elif score >= 120:
        placement = random.choice([
            'Placed',
            'Placed',
            'Not Placed',
            'Not Placed'
        ])

    else:
        placement = random.choice([
            'Not Placed',
            'Not Placed',
            'Not Placed',
            'Placed'
        ])

    student_data['Attendance'].append(attendance)
    student_data['Coding_Score'].append(coding)
    student_data['DSA_Score'].append(dsa)
    student_data['Aptitude_Score'].append(aptitude)
    student_data['CGPA'].append(cgpa)
    student_data['Study_Hours'].append(study)
    student_data['Communication_Skills'].append(communication)
    student_data['Projects'].append(projects)
    student_data['Internship'].append(internship)
    student_data['Mock_Interview'].append(mock)
    student_data['Certification_Count'].append(certifications)
    student_data['Hackathon_Participation'].append(hackathon)
    student_data['Resume_Score'].append(resume)
    student_data['Backlogs'].append(backlogs)
    student_data['Python_Score'].append(python_score)
    student_data['SQL_Score'].append(sql_score)
    student_data['Problem_Solving'].append(problem_solving)
    student_data['GitHub_Activity'].append(github)
    student_data['Placement_Status'].append(placement)

# ----------------------------------------------------------
# Create DataFrame
# ----------------------------------------------------------

df = pd.DataFrame(student_data)

print("=" * 70)
print("ADVANCED STUDENT DATASET GENERATED SUCCESSFULLY")
print("=" * 70)

print("\nFirst Five Rows\n")
print(df.head())

print("\nDataset Shape :", df.shape)

print("\nPart 1 Completed Successfully.")
# ==========================================================
# PART 2 : Exploratory Data Analysis (EDA)
# ==========================================================

print("\n" + "=" * 70)
print("PART 2 : EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# ----------------------------------------------------------
# Dataset Information
# ----------------------------------------------------------

print("\nDataset Information\n")

df.info()

# ----------------------------------------------------------
# Dataset Shape
# ----------------------------------------------------------

print("\nDataset Shape")

print(df.shape)

print(f"\nTotal Rows    : {df.shape[0]}")
print(f"Total Columns : {df.shape[1]}")

# ----------------------------------------------------------
# Column Names
# ----------------------------------------------------------

print("\nColumn Names\n")

print(df.columns.tolist())

# ----------------------------------------------------------
# First Five Rows
# ----------------------------------------------------------

print("\nFirst Five Rows\n")

print(df.head())

# ----------------------------------------------------------
# Last Five Rows
# ----------------------------------------------------------

print("\nLast Five Rows\n")

print(df.tail())

# ----------------------------------------------------------
# Statistical Summary
# ----------------------------------------------------------

print("\nStatistical Summary\n")

print(df.describe())

# ----------------------------------------------------------
# Missing Values
# ----------------------------------------------------------

print("\nMissing Values\n")

print(df.isnull().sum())

# ----------------------------------------------------------
# Duplicate Rows
# ----------------------------------------------------------

print("\nDuplicate Rows :", df.duplicated().sum())

# ----------------------------------------------------------
# Placement Distribution
# ----------------------------------------------------------

print("\nPlacement Distribution\n")

print(df["Placement_Status"].value_counts())

# ----------------------------------------------------------
# Placement Distribution Graph
# ----------------------------------------------------------

plt.figure(figsize=(6,4))

df["Placement_Status"].value_counts().plot(
    kind="bar"
)

plt.title("Placement Distribution")

plt.xlabel("Placement Status")

plt.ylabel("Number of Students")

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.show()

print("\nPart 2 Completed Successfully.")
# ==========================================================
# PART 3 : Data Preprocessing & Train-Test Split
# ==========================================================

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

print("\n" + "=" * 70)
print("PART 3 : DATA PREPROCESSING")
print("=" * 70)

# ----------------------------------------------------------
# Encode Target Variable
# ----------------------------------------------------------

encoder = LabelEncoder()

df["Placement_Status"] = encoder.fit_transform(
    df["Placement_Status"]
)

print("\nTarget Encoding Completed Successfully.")

print("\nEncoded Classes :")

for i, value in enumerate(encoder.classes_):
    print(f"{i} --> {value}")

# ----------------------------------------------------------
# Features and Target
# ----------------------------------------------------------

X = df.drop("Placement_Status", axis=1)

y = df["Placement_Status"]

print("\nFeatures Shape :", X.shape)
print("Target Shape   :", y.shape)

# ----------------------------------------------------------
# Train-Test Split
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split Completed Successfully.")

print(f"\nTraining Samples : {X_train.shape[0]}")
print(f"Testing Samples  : {X_test.shape[0]}")

print(f"\nTraining Features : {X_train.shape}")
print(f"Testing Features  : {X_test.shape}")

# ----------------------------------------------------------
# Target Distribution
# ----------------------------------------------------------

print("\nTraining Target Distribution\n")

print(y_train.value_counts())

print("\nTesting Target Distribution\n")

print(y_test.value_counts())

print("\nPart 3 Completed Successfully.")
# ==========================================================
# PART 4 : Import Machine Learning Models
# ==========================================================

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

print("\n" + "=" * 70)
print("PART 4 : MACHINE LEARNING MODELS")
print("=" * 70)

# ----------------------------------------------------------
# Create Machine Learning Models
# ----------------------------------------------------------

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42
        ),

    "Extra Trees":
        ExtraTreesClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42
        ),

    "AdaBoost":
        AdaBoostClassifier(
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            random_state=42
        ),

    "SVM":
        SVC(
            probability=True,
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            eval_metric="logloss",
            random_state=42
        ),

    "LightGBM":
        LGBMClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42
        ),

    "CatBoost":
        CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            verbose=0,
            random_state=42
        )
}

# ----------------------------------------------------------
# Verify Models
# ----------------------------------------------------------

print("\nTotal Machine Learning Models :", len(models))

print("\nAvailable Models\n")

for index, model_name in enumerate(models.keys(), start=1):
    print(f"{index}. {model_name}")

print("\nPart 4 Completed Successfully.")
# ==========================================================
# PART 5 : Automatic Model Training & Evaluation
# ==========================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score
)

import time

print("\n" + "=" * 70)
print("PART 5 : AUTOMATIC MODEL TRAINING & EVALUATION")
print("=" * 70)

# ----------------------------------------------------------
# Store Results
# ----------------------------------------------------------

results = []

# ----------------------------------------------------------
# Train & Evaluate All Models
# ----------------------------------------------------------

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(f"Training Model : {model_name}")
    print("=" * 70)

    start_time = time.time()

    model.fit(X_train, y_train)

    end_time = time.time()

    training_time = end_time - start_time

    prediction = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X_test)[:, 1]
    else:
        probability = prediction

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    precision = precision_score(
        y_test,
        prediction
    )

    recall = recall_score(
        y_test,
        prediction
    )

    f1 = f1_score(
        y_test,
        prediction
    )

    roc_auc = roc_auc_score(
        y_test,
        probability
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        prediction
    )

    results.append([

        model_name,

        accuracy,

        precision,

        recall,

        f1,

        roc_auc,

        balanced_accuracy,

        training_time

    ])

    print(f"Accuracy           : {accuracy:.4f}")
    print(f"Precision          : {precision:.4f}")
    print(f"Recall             : {recall:.4f}")
    print(f"F1 Score           : {f1:.4f}")
    print(f"ROC-AUC            : {roc_auc:.4f}")
    print(f"Balanced Accuracy  : {balanced_accuracy:.4f}")
    print(f"Training Time      : {training_time:.4f} sec")

print("\nPart 5 Completed Successfully.")
# ==========================================================
# PART 6 : Model Comparison & Best Model Selection
# ==========================================================

print("\n" + "=" * 70)
print("PART 6 : MODEL COMPARISON")
print("=" * 70)

# ----------------------------------------------------------
# Create Comparison DataFrame
# ----------------------------------------------------------

comparison = pd.DataFrame(

    results,

    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC",
        "Balanced Accuracy",
        "Training Time (sec)"
    ]
)

# ----------------------------------------------------------
# Sort Models by Accuracy
# ----------------------------------------------------------

comparison = comparison.sort_values(
    by="Accuracy",
    ascending=False
)

comparison.reset_index(
    drop=True,
    inplace=True
)

# ----------------------------------------------------------
# Round Values
# ----------------------------------------------------------

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC AUC",
    "Balanced Accuracy",
    "Training Time (sec)"
]

comparison[metric_columns] = comparison[
    metric_columns
].round(4)

# ----------------------------------------------------------
# Display Comparison Table
# ----------------------------------------------------------

print("\nMODEL COMPARISON\n")

print(comparison)

# ----------------------------------------------------------
# Save Comparison CSV
# ----------------------------------------------------------

comparison.to_csv(
    "Model_Comparison.csv",
    index=False
)

print("\nComparison CSV Saved Successfully.")

# ----------------------------------------------------------
# Top 3 Models
# ----------------------------------------------------------

print("\nTop 3 Machine Learning Models\n")

print(comparison.head(3))

# ----------------------------------------------------------
# Select Best Model
# ----------------------------------------------------------

best_model_name = comparison.iloc[0]["Model"]

best_model = models[best_model_name]

print("\nBest Model Selected :", best_model_name)

print("\nBest Model Performance\n")

print(comparison.iloc[0])

print("\nPart 6 Completed Successfully.")
# ==========================================================
# PART 7 : Model Comparison Visualization
# ==========================================================

print("\n" + "=" * 70)
print("PART 7 : MODEL COMPARISON VISUALIZATION")
print("=" * 70)

# ----------------------------------------------------------
# Accuracy Comparison
# ----------------------------------------------------------

plt.figure(figsize=(12,6))

plt.bar(
    comparison["Model"],
    comparison["Accuracy"]
)

plt.title("Accuracy Comparison of Machine Learning Models")

plt.xlabel("Machine Learning Models")

plt.ylabel("Accuracy")

plt.xticks(rotation=45, ha="right")

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.show()

# ----------------------------------------------------------
# ROC-AUC Comparison
# ----------------------------------------------------------

plt.figure(figsize=(12,6))

plt.bar(
    comparison["Model"],
    comparison["ROC AUC"]
)

plt.title("ROC-AUC Comparison of Machine Learning Models")

plt.xlabel("Machine Learning Models")

plt.ylabel("ROC-AUC Score")

plt.xticks(rotation=45, ha="right")

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.show()

# ----------------------------------------------------------
# Training Time Comparison
# ----------------------------------------------------------

plt.figure(figsize=(12,6))

plt.bar(
    comparison["Model"],
    comparison["Training Time (sec)"]
)

plt.title("Training Time Comparison")

plt.xlabel("Machine Learning Models")

plt.ylabel("Training Time (Seconds)")

plt.xticks(rotation=45, ha="right")

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.show()

print("\nPart 7 Completed Successfully.")
# ==========================================================
# PART 8 : Confusion Matrix, Classification Report &
#          Feature Importance
# ==========================================================

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

print("\n" + "=" * 70)
print("PART 8 : MODEL EVALUATION")
print("=" * 70)

# ----------------------------------------------------------
# Prediction Using Best Model
# ----------------------------------------------------------

prediction = best_model.predict(X_test)

# ----------------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------------

cm = confusion_matrix(
    y_test,
    prediction
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

display.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.tight_layout()

plt.show()

# ----------------------------------------------------------
# Classification Report
# ----------------------------------------------------------

print("\nClassification Report\n")

print(classification_report(
    y_test,
    prediction
))

# ----------------------------------------------------------
# Feature Importance
# ----------------------------------------------------------

if hasattr(best_model, "feature_importances_"):

    feature_df = pd.DataFrame({

        "Feature": X.columns,

        "Importance": best_model.feature_importances_

    })

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=False
    )

elif hasattr(best_model, "coef_"):

    feature_df = pd.DataFrame({

        "Feature": X.columns,

        "Importance": abs(best_model.coef_[0])

    })

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=False
    )

else:

    feature_df = None

# ----------------------------------------------------------
# Display Feature Importance
# ----------------------------------------------------------

if feature_df is not None:

    print("\nFeature Importance\n")

    print(feature_df)

    plt.figure(figsize=(10,7))

    plt.barh(
        feature_df["Feature"],
        feature_df["Importance"]
    )

    plt.title("Feature Importance")

    plt.xlabel("Importance Score")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.show()

else:

    print("\nFeature Importance is not available for this model.")

print("\nPart 8 Completed Successfully.")
# ==========================================================
# PART 9 : Save Model, Load Model & Prediction System
# ==========================================================

import pickle
import os

print("\n" + "=" * 70)
print("PART 9 : MODEL SERIALIZATION & PREDICTION")
print("=" * 70)

# ----------------------------------------------------------
# Save Best Model
# ----------------------------------------------------------

with open("best_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("\nBest Model Saved Successfully.")

# ----------------------------------------------------------
# Save Label Encoder
# ----------------------------------------------------------

with open("label_encoder.pkl", "wb") as file:
    pickle.dump(encoder, file)

print("Label Encoder Saved Successfully.")

# ----------------------------------------------------------
# Verify Saved Files
# ----------------------------------------------------------

print("\nVerification")

print("Best Model Exists :", os.path.exists("best_model.pkl"))

print("Encoder Exists    :", os.path.exists("label_encoder.pkl"))

# ----------------------------------------------------------
# Load Saved Model
# ----------------------------------------------------------

with open("best_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

with open("label_encoder.pkl", "rb") as file:
    loaded_encoder = pickle.load(file)

print("\nSaved Model Loaded Successfully.")

# ----------------------------------------------------------
# New Student Data
# ----------------------------------------------------------

new_student = pd.DataFrame({

    "Attendance":[90],
    "Coding_Score":[88],
    "DSA_Score":[84],
    "Aptitude_Score":[82],
    "CGPA":[8.7],
    "Study_Hours":[8],
    "Communication_Skills":[8],
    "Projects":[4],
    "Internship":[1],
    "Mock_Interview":[8],
    "Certification_Count":[5],
    "Hackathon_Participation":[2],
    "Resume_Score":[88],
    "Backlogs":[0],
    "Python_Score":[90],
    "SQL_Score":[85],
    "Problem_Solving":[9],
    "GitHub_Activity":[80]

})

# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

prediction = loaded_model.predict(new_student)

result = loaded_encoder.inverse_transform(prediction)

print("\nPrediction Result")

print("Placement Status :", result[0])

# ----------------------------------------------------------
# Prediction Probability
# ----------------------------------------------------------

if hasattr(loaded_model, "predict_proba"):

    probability = loaded_model.predict_proba(new_student)

    placed_index = list(loaded_model.classes_).index(1)

    placement_probability = probability[0][placed_index]

    print(
        "Placement Probability :",
        f"{placement_probability * 100:.2f}%"
    )

# ----------------------------------------------------------
# Project Summary
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)

print(f"Dataset Shape           : {df.shape}")

print(f"Total Models Compared   : {len(models)}")

print(f"Best Model              : {best_model_name}")

print("\nTop 3 Models")

print(comparison.head(3))

print("\nProject Completed Successfully.")

print("=" * 70)