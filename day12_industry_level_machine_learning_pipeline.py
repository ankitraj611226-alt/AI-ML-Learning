# ==========================================================
# PROJECT : Industry Level Machine Learning Pipeline
# FILE : day12_industry_level_machine_learning_pipeline.py
# PART 1 : Import Libraries & Generate Dataset
# ==========================================================

# ----------------------------------------------------------
# Import Required Libraries
# ----------------------------------------------------------

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import os

from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler
)

from sklearn.pipeline import Pipeline

from sklearn.feature_selection import (
    SelectKBest,
    f_classif
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc
)

warnings.filterwarnings("ignore")

# ----------------------------------------------------------
# Set Random Seed
# ----------------------------------------------------------

random.seed(42)
np.random.seed(42)

# ----------------------------------------------------------
# Generate Student Placement Dataset
# ----------------------------------------------------------

student_data = {

    "Attendance": np.random.randint(60,101,5000),
    "Coding_Score": np.random.randint(40,101,5000),
    "DSA_Score": np.random.randint(35,101,5000),
    "Python_Score": np.random.randint(40,101,5000),
    "SQL_Score": np.random.randint(35,101,5000),
    "CGPA": np.round(np.random.uniform(5.5,9.8,5000),2),
    "Study_Hours": np.random.randint(1,8,5000),
    "Communication_Skills": np.random.randint(4,11,5000),
    "Projects": np.random.randint(0,8,5000),
    "Internship": np.random.randint(0,2,5000),
    "Mock_Interview": np.random.randint(4,11,5000),
    "Certification_Count": np.random.randint(0,6,5000),
    "Hackathon_Participation": np.random.randint(0,6,5000),
    "GitHub_Activity": np.random.randint(0,101,5000),
    "Resume_Score": np.random.randint(40,101,5000)

}

df = pd.DataFrame(student_data)

# ----------------------------------------------------------
# Generate Placement Status
# ----------------------------------------------------------

placement_score = (

    df["Coding_Score"] * 0.18 +
    df["DSA_Score"] * 0.15 +
    df["Python_Score"] * 0.12 +
    df["SQL_Score"] * 0.10 +
    df["CGPA"] * 10 * 0.12 +
    df["Communication_Skills"] * 10 * 0.10 +
    df["Projects"] * 5 +
    df["Internship"] * 15 +
    df["Certification_Count"] * 3 +
    df["GitHub_Activity"] * 0.05

)

df["Placement_Status"] = np.where(
    placement_score >= 85,
    1,
    0
)

# ----------------------------------------------------------
# Dataset Preview
# ----------------------------------------------------------

print("=" * 70)
print("PART 1 : DATASET GENERATED SUCCESSFULLY")
print("=" * 70)

print("\nFirst Five Rows\n")

print(df.head())

print("\nDataset Shape :", df.shape)

print("\nPart 1 Completed Successfully.")
# ==========================================================
# PART 2 : Data Preprocessing & Train-Test Split
# ==========================================================

print("\n" + "=" * 70)
print("PART 2 : DATA PREPROCESSING")
print("=" * 70)

# ----------------------------------------------------------
# Dataset Information
# ----------------------------------------------------------

print("\nDataset Information\n")

df.info()

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
# Features and Target
# ----------------------------------------------------------

X = df.drop(
    "Placement_Status",
    axis=1
)

y = df["Placement_Status"]

print("\nFeatures Shape :", X.shape)
print("Target Shape   :", y.shape)

# ----------------------------------------------------------
# Train Test Split
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

print(f"\nTraining Shape : {X_train.shape}")
print(f"Testing Shape  : {X_test.shape}")

# ----------------------------------------------------------
# Machine Learning Models
# ----------------------------------------------------------

models = {

    "Logistic Regression":
    LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree":
    DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest":
    RandomForestClassifier(
        random_state=42
    ),

    "Gradient Boosting":
    GradientBoostingClassifier(
        random_state=42
    ),

    "KNN":
    KNeighborsClassifier(),

    "SVM":
    SVC(
        probability=True,
        random_state=42
    )

}

print("\nDataset Ready Successfully.")

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

print("Total Models :", len(models))

print("\nPart 2 Completed Successfully.")
# ==========================================================
# PART 3 : Cross Validation & Model Comparison
# ==========================================================

print("\n" + "=" * 70)
print("PART 3 : CROSS VALIDATION")
print("=" * 70)

# ----------------------------------------------------------
# K-Fold Cross Validation
# ----------------------------------------------------------

kfold = KFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)

print("\nK-Fold Configuration\n")

print(kfold)

# ----------------------------------------------------------
# Logistic Regression Cross Validation
# ----------------------------------------------------------

logistic_model = LogisticRegression(
    max_iter=1000
)

kfold_scores = cross_val_score(

    logistic_model,

    X,

    y,

    cv=kfold,

    scoring="accuracy"

)

print("\nLogistic Regression")

print("Fold Scores :")

print(kfold_scores)

print("\nAverage Accuracy :", round(kfold_scores.mean(),4))

# ----------------------------------------------------------
# Stratified K-Fold
# ----------------------------------------------------------

stratified = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)

print("\nStratified K-Fold\n")

print(stratified)

# ----------------------------------------------------------
# Random Forest Cross Validation
# ----------------------------------------------------------

rf_model = RandomForestClassifier(
    random_state=42
)

cv_scores = cross_val_score(

    rf_model,

    X,

    y,

    cv=stratified,

    scoring="accuracy"

)

print("\nRandom Forest")

print("Cross Validation Scores :")

print(cv_scores)

print("\nMean Accuracy :", round(cv_scores.mean(),4))

print("Standard Deviation :", round(cv_scores.std(),4))

# ----------------------------------------------------------
# Compare All Models
# ----------------------------------------------------------

cv_results = []

for model_name, model in models.items():

    scores = cross_val_score(

        model,

        X,

        y,

        cv=5,

        scoring="accuracy"

    )

    cv_results.append({

        "Model": model_name,

        "Average CV Accuracy": round(scores.mean(),4),

        "Standard Deviation": round(scores.std(),4)

    })

# ----------------------------------------------------------
# Comparison DataFrame
# ----------------------------------------------------------

cv_df = pd.DataFrame(cv_results)

cv_df = cv_df.sort_values(

    by="Average CV Accuracy",

    ascending=False

).reset_index(drop=True)

print("\n" + "=" * 70)
print("CROSS VALIDATION RESULTS")
print("=" * 70)

print(cv_df)

# ----------------------------------------------------------
# Save Cross Validation Report
# ----------------------------------------------------------

cv_df.to_csv(
    "Cross_Validation_Report.csv",
    index=False
)

print("\nCross Validation Report Saved Successfully.")

# ----------------------------------------------------------
# Best Model
# ----------------------------------------------------------

best_model_name = cv_df.iloc[0]["Model"]

print("\nBest Model :", best_model_name)

print("\nPart 3 Completed Successfully.")
# ==========================================================
# PART 4 : Overfitting, Underfitting & Error Analysis
# ==========================================================

from sklearn.model_selection import learning_curve

print("\n" + "=" * 70)
print("PART 4 : ERROR ANALYSIS & LEARNING CURVE")
print("=" * 70)

# ----------------------------------------------------------
# Train Random Forest
# ----------------------------------------------------------

rf = RandomForestClassifier(
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

# ----------------------------------------------------------
# Training & Testing Prediction
# ----------------------------------------------------------

train_prediction = rf.predict(X_train)

test_prediction = rf.predict(X_test)

# ----------------------------------------------------------
# Training Accuracy
# ----------------------------------------------------------

train_accuracy = accuracy_score(
    y_train,
    train_prediction
)

test_accuracy = accuracy_score(
    y_test,
    test_prediction
)

print("\nTraining Accuracy :", round(train_accuracy,4))

print("Testing Accuracy  :", round(test_accuracy,4))

# ----------------------------------------------------------
# Overfitting Check
# ----------------------------------------------------------

if train_accuracy - test_accuracy > 0.05:

    print("\nModel Status : Overfitting Detected")

elif train_accuracy < 0.70:

    print("\nModel Status : Underfitting Detected")

else:

    print("\nModel Status : Good Generalization")

# ----------------------------------------------------------
# Error Analysis
# ----------------------------------------------------------

comparison_prediction = X_test.copy()

comparison_prediction["Actual"] = y_test.values

comparison_prediction["Prediction"] = test_prediction

comparison_prediction["Correct"] = (

    comparison_prediction["Actual"]

    ==

    comparison_prediction["Prediction"]

)

print("\nPrediction Comparison\n")

print(comparison_prediction.head())

# ----------------------------------------------------------
# Wrong Predictions
# ----------------------------------------------------------

wrong_predictions = comparison_prediction[
    comparison_prediction["Correct"] == False
]

print("\nTotal Wrong Predictions :", len(wrong_predictions))

wrong_predictions.to_csv(
    "Wrong_Predictions.csv",
    index=False
)

print("Wrong Prediction Report Saved Successfully.")

# ----------------------------------------------------------
# Learning Curve
# ----------------------------------------------------------

train_sizes, train_scores, test_scores = learning_curve(

    RandomForestClassifier(random_state=42),

    X,

    y,

    cv=5,

    scoring="accuracy",

    train_sizes=np.linspace(
        0.1,
        1.0,
        5
    )

)

train_mean = train_scores.mean(axis=1)

test_mean = test_scores.mean(axis=1)

plt.figure(figsize=(8,5))

plt.plot(

    train_sizes,

    train_mean,

    marker="o",

    label="Training Accuracy"

)

plt.plot(

    train_sizes,

    test_mean,

    marker="o",

    label="Validation Accuracy"

)

plt.title("Learning Curve")

plt.xlabel("Training Samples")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

print("\nLearning Curve Generated Successfully.")

print("\nPart 4 Completed Successfully.")
# ==========================================================
# PART 5 : Hyperparameter Tuning
# ==========================================================

print("\n" + "=" * 70)
print("PART 5 : HYPERPARAMETER TUNING")
print("=" * 70)

# ----------------------------------------------------------
# Parameter Grid
# ----------------------------------------------------------

parameter_grid = {

    "n_estimators": [100, 200, 300],

    "max_depth": [5, 10, 15],

    "min_samples_split": [2, 5],

    "min_samples_leaf": [1, 2]

}

# ----------------------------------------------------------
# Grid Search CV
# ----------------------------------------------------------

grid_search = GridSearchCV(

    estimator=RandomForestClassifier(random_state=42),

    param_grid=parameter_grid,

    cv=5,

    scoring="accuracy",

    n_jobs=-1

)

grid_search.fit(

    X_train,

    y_train

)

print("\nGrid Search Completed Successfully.")

print("\nBest Parameters")

print(grid_search.best_params_)

print("\nBest Cross Validation Score")

print(round(grid_search.best_score_,4))

# ----------------------------------------------------------
# Best Model
# ----------------------------------------------------------

best_rf = grid_search.best_estimator_

prediction = best_rf.predict(X_test)

grid_accuracy = accuracy_score(

    y_test,

    prediction

)

print("\nTesting Accuracy :", round(grid_accuracy,4))

# ----------------------------------------------------------
# Randomized Search CV
# ----------------------------------------------------------

random_search = RandomizedSearchCV(

    estimator=RandomForestClassifier(random_state=42),

    param_distributions=parameter_grid,

    n_iter=10,

    cv=5,

    scoring="accuracy",

    random_state=42,

    n_jobs=-1

)

random_search.fit(

    X_train,

    y_train

)

print("\nRandomized Search Completed Successfully.")

# ----------------------------------------------------------
# Compare Grid Search & Random Search
# ----------------------------------------------------------

comparison_search = pd.DataFrame({

    "Method": [

        "GridSearchCV",

        "RandomizedSearchCV"

    ],

    "Best Score": [

        round(grid_search.best_score_,4),

        round(random_search.best_score_,4)

    ]

})

print("\nHyperparameter Tuning Comparison\n")

print(comparison_search)

# ----------------------------------------------------------
# Save Comparison Report
# ----------------------------------------------------------

comparison_search.to_csv(

    "Hyperparameter_Tuning_Report.csv",

    index=False

)

print("\nHyperparameter Tuning Report Saved Successfully.")

print("\nPart 5 Completed Successfully.")
# ==========================================================
# PART 6 : Feature Engineering & Feature Selection
# ==========================================================

print("\n" + "=" * 70)
print("PART 6 : FEATURE ENGINEERING & FEATURE SELECTION")
print("=" * 70)

# ----------------------------------------------------------
# Create New Features
# ----------------------------------------------------------

df["Technical_Score"] = (

    df["Coding_Score"] +

    df["DSA_Score"] +

    df["Python_Score"] +

    df["SQL_Score"]

) / 4

df["Profile_Score"] = (

    df["Projects"] * 10 +

    df["Certification_Count"] * 5 +

    df["Hackathon_Participation"] * 8 +

    df["Resume_Score"]

) / 4

df["Interview_Score"] = (

    df["Communication_Skills"] * 10 +

    df["Mock_Interview"] * 10

) / 2

print("\nNew Features Created Successfully.")

print(df[[
    "Technical_Score",
    "Profile_Score",
    "Interview_Score"
]].head())

# ----------------------------------------------------------
# Update Features & Target
# ----------------------------------------------------------

X = df.drop(
    "Placement_Status",
    axis=1
)

y = df["Placement_Status"]

print("\nUpdated Feature Count :", X.shape[1])

# ----------------------------------------------------------
# Feature Selection
# ----------------------------------------------------------

selector = SelectKBest(

    score_func=f_classif,

    k=10

)

X_selected = selector.fit_transform(
    X,
    y
)

print("\nOriginal Features :", X.shape[1])

print("Selected Features :", X_selected.shape[1])

# ----------------------------------------------------------
# Best Features
# ----------------------------------------------------------

selected_features = X.columns[
    selector.get_support()
]

print("\nTop Selected Features\n")

for feature in selected_features:

    print(feature)

# ----------------------------------------------------------
# Feature Scores
# ----------------------------------------------------------

feature_scores = pd.DataFrame({

    "Feature": X.columns,

    "Score": selector.scores_

})

feature_scores = feature_scores.sort_values(

    by="Score",

    ascending=False

)

print("\nTop 15 Important Features\n")

print(feature_scores.head(15))

# ----------------------------------------------------------
# Save Feature Scores
# ----------------------------------------------------------

feature_scores.to_csv(

    "Feature_Selection_Report.csv",

    index=False

)

print("\nFeature Selection Report Saved Successfully.")

# ----------------------------------------------------------
# Feature Importance Graph
# ----------------------------------------------------------

plt.figure(figsize=(10,6))

plt.barh(

    feature_scores["Feature"][:10],

    feature_scores["Score"][:10]

)

plt.title("Top Selected Features")

plt.xlabel("ANOVA Score")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()

print("\nPart 6 Completed Successfully.")
# ==========================================================
# PART 7 : Machine Learning Pipeline
# ==========================================================

print("\n" + "=" * 70)
print("PART 7 : MACHINE LEARNING PIPELINE")
print("=" * 70)

# ----------------------------------------------------------
# Train Test Split (Updated Features)
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

# ----------------------------------------------------------
# Create Machine Learning Pipeline
# ----------------------------------------------------------

pipeline = Pipeline([

    (

        "scaler",

        StandardScaler()

    ),

    (

        "feature_selection",

        SelectKBest(

            score_func=f_classif,

            k=10

        )

    ),

    (

        "classifier",

        RandomForestClassifier(

            random_state=42

        )

    )

])

print("\nMachine Learning Pipeline Created Successfully.")

print("\nPipeline Structure\n")

print(pipeline)

# ----------------------------------------------------------
# Train Pipeline
# ----------------------------------------------------------

pipeline.fit(

    X_train,

    y_train

)

print("\nPipeline Training Completed Successfully.")

# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

pipeline_prediction = pipeline.predict(

    X_test

)

# ----------------------------------------------------------
# Evaluation
# ----------------------------------------------------------

pipeline_accuracy = accuracy_score(

    y_test,

    pipeline_prediction

)

pipeline_precision = precision_score(

    y_test,

    pipeline_prediction

)

pipeline_recall = recall_score(

    y_test,

    pipeline_prediction

)

pipeline_f1 = f1_score(

    y_test,

    pipeline_prediction

)

print("\nPipeline Performance")

print(f"\nAccuracy  : {pipeline_accuracy:.4f}")

print(f"Precision : {pipeline_precision:.4f}")

print(f"Recall    : {pipeline_recall:.4f}")

print(f"F1 Score  : {pipeline_f1:.4f}")

# ----------------------------------------------------------
# Cross Validation
# ----------------------------------------------------------

pipeline_scores = cross_val_score(

    pipeline,

    X,

    y,

    cv=5,

    scoring="accuracy"

)

print("\nCross Validation Scores")

print(pipeline_scores)

print(

    "\nAverage Accuracy :",

    round(

        pipeline_scores.mean(),

        4

    )

)

# ----------------------------------------------------------
# Save Pipeline Performance
# ----------------------------------------------------------

pipeline_report = pd.DataFrame({

    "Metric":[

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "Average Cross Validation"

    ],

    "Value":[

        pipeline_accuracy,

        pipeline_precision,

        pipeline_recall,

        pipeline_f1,

        pipeline_scores.mean()

    ]

})

pipeline_report.to_csv(

    "Pipeline_Report.csv",

    index=False

)

print("\nPipeline Report Saved Successfully.")

print("\nPart 7 Completed Successfully.")
# ==========================================================
# PART 8 : Pipeline Serialization & Explainable AI
# ==========================================================

print("\n" + "=" * 70)
print("PART 8 : PIPELINE SERIALIZATION & EXPLAINABLE AI")
print("=" * 70)

# ----------------------------------------------------------
# Save Complete Pipeline
# ----------------------------------------------------------

with open("placement_pipeline.pkl", "wb") as file:

    pickle.dump(
        pipeline,
        file
    )

print("\nPipeline Saved Successfully.")

# ----------------------------------------------------------
# Verify Pipeline
# ----------------------------------------------------------

print("\nPipeline Exists :", os.path.exists("placement_pipeline.pkl"))

# ----------------------------------------------------------
# Load Pipeline
# ----------------------------------------------------------

with open("placement_pipeline.pkl", "rb") as file:

    loaded_pipeline = pickle.load(file)

print("\nPipeline Loaded Successfully.")

# ----------------------------------------------------------
# New Student
# ----------------------------------------------------------

new_student = pd.DataFrame({

    "Attendance":[90],
    "Coding_Score":[85],
    "DSA_Score":[80],
    "Python_Score":[88],
    "SQL_Score":[82],
    "CGPA":[8.4],
    "Study_Hours":[5],
    "Communication_Skills":[8],
    "Projects":[4],
    "Internship":[1],
    "Mock_Interview":[8],
    "Certification_Count":[3],
    "Hackathon_Participation":[2],
    "GitHub_Activity":[75],
    "Resume_Score":[85],
    "Technical_Score":[(85+80+88+82)/4],
    "Profile_Score":[((4*10)+(3*5)+(2*8)+85)/4],
    "Interview_Score":[((8*10)+(8*10))/2]

})

# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

prediction = loaded_pipeline.predict(
    new_student
)

probability = loaded_pipeline.predict_proba(
    new_student
)

print("\nPrediction Result")

if prediction[0] == 1:

    print("Placement Status : PLACED")

else:

    print("Placement Status : NOT PLACED")

print("\nPrediction Probability")

print(probability)

# ----------------------------------------------------------
# Confidence Score
# ----------------------------------------------------------

confidence = np.max(
    probability
) * 100

print(f"\nModel Confidence : {confidence:.2f}%")

# ----------------------------------------------------------
# AI Career Recommendation Engine
# ----------------------------------------------------------

recommendations = []

student = new_student.iloc[0]

if student["Coding_Score"] < 70:

    recommendations.append(
        "Improve Coding Skills"
    )

if student["DSA_Score"] < 70:

    recommendations.append(
        "Practice DSA Daily"
    )

if student["Communication_Skills"] < 7:

    recommendations.append(
        "Improve Communication Skills"
    )

if student["Projects"] < 3:

    recommendations.append(
        "Build More Projects"
    )

if student["Certification_Count"] < 3:

    recommendations.append(
        "Complete Industry Certifications"
    )

if student["Internship"] == 0:

    recommendations.append(
        "Apply for Internship"
    )

if student["GitHub_Activity"] < 50:

    recommendations.append(
        "Become Active on GitHub"
    )

if len(recommendations) == 0:

    recommendations.append(
        "Excellent Profile. Keep Improving."
    )

print("\nAI Career Recommendations\n")

for item in recommendations:

    print("-", item)

# ----------------------------------------------------------
# AI Decision Engine
# ----------------------------------------------------------

if confidence >= 95:

    confidence_level = "Very High"

elif confidence >= 85:

    confidence_level = "High"

elif confidence >= 70:

    confidence_level = "Medium"

else:

    confidence_level = "Low"

print("\nConfidence Level :", confidence_level)

# ----------------------------------------------------------
# Generate AI Placement Report
# ----------------------------------------------------------

report = {

    "Prediction":
    "Placed" if prediction[0] == 1 else "Not Placed",

    "Confidence (%)":
    round(confidence,2),

    "Technical Score":
    student["Technical_Score"],

    "Profile Score":
    student["Profile_Score"],

    "Interview Score":
    student["Interview_Score"],

    "Recommendations":
    ", ".join(recommendations)

}

report_df = pd.DataFrame([report])

report_df.to_csv(

    "AI_Placement_Report.csv",

    index=False

)

print("\nAI Placement Report Saved Successfully.")

print("\nPart 8 Completed Successfully.")
# ==========================================================
# PART 9 : AI Performance Dashboard & Project Summary
# ==========================================================

print("\n" + "=" * 70)
print("PART 9 : AI PERFORMANCE DASHBOARD")
print("=" * 70)

# ----------------------------------------------------------
# Prediction Probability
# ----------------------------------------------------------

probability = pipeline.predict_proba(X_test)[:, 1]

# ----------------------------------------------------------
# ROC Curve
# ----------------------------------------------------------

fpr, tpr, threshold = roc_curve(
    y_test,
    probability
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure(figsize=(7,6))

plt.plot(
    fpr,
    tpr,
    label=f"ROC Curve (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0,1],
    [0,1],
    "--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# ----------------------------------------------------------
# Precision Recall Curve
# ----------------------------------------------------------

precision_curve, recall_curve, threshold = precision_recall_curve(
    y_test,
    probability
)

plt.figure(figsize=(7,6))

plt.plot(
    recall_curve,
    precision_curve
)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision Recall Curve")

plt.grid(True)

plt.tight_layout()

plt.show()

# ----------------------------------------------------------
# AI Performance Dashboard
# ----------------------------------------------------------

dashboard = pd.DataFrame({

    "Metric":[

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "ROC AUC",

        "Confidence Score"

    ],

    "Value":[

        round(pipeline_accuracy,4),

        round(pipeline_precision,4),

        round(pipeline_recall,4),

        round(pipeline_f1,4),

        round(roc_auc,4),

        round(confidence,2)

    ]

})

print("\nAI Performance Dashboard\n")

print(dashboard)

# ----------------------------------------------------------
# Save Dashboard
# ----------------------------------------------------------

dashboard.to_csv(

    "AI_Model_Report.csv",

    index=False

)

print("\nAI Model Report Saved Successfully.")

# ----------------------------------------------------------
# Project Summary
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)

print(f"Dataset Shape              : {df.shape}")

print(f"Training Samples           : {X_train.shape[0]}")

print(f"Testing Samples            : {X_test.shape[0]}")

print(f"Total Features             : {X.shape[1]}")

print(f"Pipeline Accuracy          : {pipeline_accuracy:.4f}")

print(f"Pipeline Precision         : {pipeline_precision:.4f}")

print(f"Pipeline Recall            : {pipeline_recall:.4f}")

print(f"Pipeline F1 Score          : {pipeline_f1:.4f}")

print(f"Cross Validation Accuracy  : {pipeline_scores.mean():.4f}")

print(f"ROC AUC Score              : {roc_auc:.4f}")

print(f"Confidence Score           : {confidence:.2f}%")

print("\nGenerated Files")

print("- Cross_Validation_Report.csv")
print("- Wrong_Predictions.csv")
print("- Hyperparameter_Tuning_Report.csv")
print("- Feature_Selection_Report.csv")
print("- Pipeline_Report.csv")
print("- AI_Placement_Report.csv")
print("- AI_Model_Report.csv")
print("- placement_pipeline.pkl")

print("\nProject Completed Successfully.")

print("=" * 70)