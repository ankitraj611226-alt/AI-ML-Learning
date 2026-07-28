"""
=========================================================
          Day 08 - Hyperparameter Tuning
               & Cross Validation
---------------------------------------------------------
Author      : Ankit Raj
Language    : Python
Description :
Improve a Student Placement Prediction Model using
Cross Validation and Hyperparameter Tuning.

Topics Covered:
✔ Dataset Generation
✔ Label Encoding
✔ Train-Test Split
✔ KNN Classifier
✔ Random Forest Classifier
✔ Cross Validation
✔ GridSearchCV
✔ Hyperparameter Tuning
✔ Placement Prediction
✔ Placement Probability
=========================================================
"""

# =========================================================
# Import Libraries
# =========================================================

import random
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# Set Random Seed
# ---------------------------------------------------------

random.seed(42)

print("=" * 70)
print("DAY 08 - HYPERPARAMETER TUNING & CROSS VALIDATION")
print("=" * 70)

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
                "Placed",
                "Not Placed",
                "Not Placed"
            ]
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

print("\nDataset Shape :", df.shape)

print("\nFirst Five Records\n")
print(df.head())

print("\nDataset Information\n")
print(df.info())

print("\nStatistical Summary\n")
print(df.describe())
# =========================================================
# Label Encoding
# =========================================================

print("\n" + "=" * 70)
print("LABEL ENCODING")
print("=" * 70)

encoder = LabelEncoder()

df["Placement_Status"] = encoder.fit_transform(
    df["Placement_Status"]
)

print("\nEncoded Dataset\n")
print(df.head())

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

print("\nFeature Matrix Shape :", X.shape)
print("Target Vector Shape  :", y.shape)

# =========================================================
# Train-Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# =========================================================
# Baseline KNN Model
# =========================================================

print("\n" + "=" * 70)
print("BASELINE KNN MODEL")
print("=" * 70)

knn_model = KNeighborsClassifier()

knn_model.fit(X_train, y_train)

knn_predictions = knn_model.predict(X_test)

knn_accuracy = accuracy_score(
    y_test,
    knn_predictions
)

print("Baseline KNN Accuracy :", round(knn_accuracy, 4))

# =========================================================
# Baseline Random Forest Model
# =========================================================

print("\n" + "=" * 70)
print("BASELINE RANDOM FOREST MODEL")
print("=" * 70)

rf_model = RandomForestClassifier(
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

print("Baseline Random Forest Accuracy :", round(rf_accuracy, 4))

# =========================================================
# Compare Baseline Models
# =========================================================

comparison_df = pd.DataFrame({

    "Model": [
        "KNN",
        "Random Forest"
    ],

    "Accuracy": [
        knn_accuracy,
        rf_accuracy
    ]

})

print("\nBaseline Model Comparison\n")
print(comparison_df)

# =========================================================
# Visualize Baseline Performance
# =========================================================

plt.figure(figsize=(7,5))

bars = plt.bar(
    comparison_df["Model"],
    comparison_df["Accuracy"]
)

plt.title("Baseline Model Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

# Display accuracy value above each bar

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.01,
        f"{height:.2f}",
        ha="center",
        fontsize=10
    )

plt.tight_layout()
plt.show()
# =========================================================
# Cross Validation
# =========================================================

print("\n" + "=" * 70)
print("5-FOLD CROSS VALIDATION")
print("=" * 70)

# Perform Cross Validation using Random Forest

cv_scores = cross_val_score(
    estimator=rf_model,
    X=X,
    y=y,
    cv=5,
    scoring="accuracy"
)

# =========================================================
# Display Cross Validation Scores
# =========================================================

print("\nCross Validation Scores")

for fold, score in enumerate(cv_scores, start=1):
    print(f"Fold {fold} Accuracy : {score:.4f}")

# =========================================================
# Cross Validation Statistics
# =========================================================

average_score = cv_scores.mean()
minimum_score = cv_scores.min()
maximum_score = cv_scores.max()
standard_deviation = cv_scores.std()

print("\nCross Validation Summary")
print("-" * 40)

print(f"Average Accuracy       : {average_score:.4f}")
print(f"Highest Accuracy       : {maximum_score:.4f}")
print(f"Lowest Accuracy        : {minimum_score:.4f}")
print(f"Standard Deviation     : {standard_deviation:.4f}")

# =========================================================
# Visualize Cross Validation Scores
# =========================================================

cv_results = pd.DataFrame({
    "Fold": [1, 2, 3, 4, 5],
    "Accuracy": cv_scores
})

plt.figure(figsize=(8,5))

bars = plt.bar(
    cv_results["Fold"].astype(str),
    cv_results["Accuracy"]
)

plt.title("5-Fold Cross Validation Accuracy")
plt.xlabel("Fold Number")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

# Show Accuracy Above Bars

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.01,
        f"{height:.3f}",
        ha="center",
        fontsize=10
    )

plt.tight_layout()
plt.show()

# =========================================================
# Cross Validation Analysis
# =========================================================

print("\nCross Validation Analysis")
print("-" * 40)

if standard_deviation < 0.03:
    print("The model is stable and generalizes well.")
elif standard_deviation < 0.07:
    print("The model is reasonably stable.")
else:
    print("The model may be unstable and could require improvement.")

print("\nCross Validation Completed Successfully!")
# =========================================================
# Hyperparameter Tuning - KNN
# =========================================================

print("\n" + "=" * 70)
print("KNN HYPERPARAMETER TUNING")
print("=" * 70)

knn_param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11, 13]
}

knn_grid = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=knn_param_grid,
    cv=5,
    scoring="accuracy"
)

knn_grid.fit(X_train, y_train)

# Best KNN Model

best_knn_model = knn_grid.best_estimator_

# Prediction

knn_tuned_predictions = best_knn_model.predict(X_test)

# Accuracy

knn_tuned_accuracy = accuracy_score(
    y_test,
    knn_tuned_predictions
)

print("\nBest K Value")
print(knn_grid.best_params_)

print(
    "Best Cross Validation Accuracy :",
    round(knn_grid.best_score_, 4)
)

print(
    "Tuned KNN Test Accuracy :",
    round(knn_tuned_accuracy, 4)
)

# =========================================================
# Random Forest Hyperparameter Tuning
# =========================================================

print("\n" + "=" * 70)
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 70)

rf_param_grid = {

    "n_estimators": [50, 100, 200],

    "max_depth": [3, 5, 10, None],

    "min_samples_split": [2, 5, 10]

}

rf_grid = GridSearchCV(

    estimator=RandomForestClassifier(
        random_state=42
    ),

    param_grid=rf_param_grid,

    cv=5,

    scoring="accuracy",

    n_jobs=-1

)

rf_grid.fit(X_train, y_train)

# Best Random Forest Model

best_rf_model = rf_grid.best_estimator_

# Prediction

rf_tuned_predictions = best_rf_model.predict(
    X_test
)

# Accuracy

rf_tuned_accuracy = accuracy_score(
    y_test,
    rf_tuned_predictions
)

print("\nBest Parameters")
print(rf_grid.best_params_)

print(
    "Best Cross Validation Accuracy :",
    round(rf_grid.best_score_, 4)
)

print(
    "Tuned Random Forest Accuracy :",
    round(rf_tuned_accuracy, 4)
)

# =========================================================
# Accuracy Comparison
# =========================================================

comparison = pd.DataFrame({

    "Model": [

        "Baseline KNN",
        "Tuned KNN",
        "Baseline RF",
        "Tuned RF"

    ],

    "Accuracy": [

        knn_accuracy,
        knn_tuned_accuracy,
        rf_accuracy,
        rf_tuned_accuracy

    ]

})

print("\nModel Comparison\n")
print(comparison)

# =========================================================
# Visualization
# =========================================================

plt.figure(figsize=(9,5))

bars = plt.bar(

    comparison["Model"],
    comparison["Accuracy"]

)

plt.title("Model Accuracy Before & After Hyperparameter Tuning")

plt.xlabel("Models")

plt.ylabel("Accuracy")

plt.ylim(0,1)

# Accuracy Labels

for bar in bars:

    height = bar.get_height()

    plt.text(

        bar.get_x() + bar.get_width()/2,

        height + 0.01,

        f"{height:.3f}",

        ha="center",

        fontsize=10

    )

plt.tight_layout()

plt.show()

# =========================================================
# Improvement Analysis
# =========================================================

print("\nImprovement Summary")
print("-" * 40)

print(
    f"KNN Improvement : "
    f"{(knn_tuned_accuracy - knn_accuracy):.4f}"
)

print(
    f"Random Forest Improvement : "
    f"{(rf_tuned_accuracy - rf_accuracy):.4f}"
)

print("\nHyperparameter Tuning Completed Successfully!")
# =========================================================
# Predict New Student
# =========================================================

print("\n" + "=" * 70)
print("NEW STUDENT PLACEMENT PREDICTION")
print("=" * 70)

new_student = pd.DataFrame(
    {
        "Attendance": [85],
        "Coding_Score": [78],
        "Study_Hours": [7],
        "Communication_Skills": [8]
    }
)

prediction = best_rf_model.predict(new_student)

prediction_result = encoder.inverse_transform(prediction)

print("\nStudent Details")
print(new_student)

print("\nPrediction :", prediction_result[0])

# =========================================================
# Placement Probability
# =========================================================

probability = best_rf_model.predict_proba(new_student)

placed_index = list(best_rf_model.classes_).index(1)

placement_probability = probability[0][placed_index]

print(
    "Placement Probability :",
    f"{placement_probability * 100:.2f}%"
)

# =========================================================
# Predict Multiple Students
# =========================================================

print("\n" + "=" * 70)
print("MULTIPLE STUDENT PREDICTION")
print("=" * 70)

students = pd.DataFrame({

    "Attendance": [95, 60, 75, 88],

    "Coding_Score": [90, 40, 65, 82],

    "Study_Hours": [8, 3, 6, 7],

    "Communication_Skills": [9, 4, 6, 8]

})

predictions = best_rf_model.predict(students)

students["Prediction"] = encoder.inverse_transform(predictions)

probabilities = best_rf_model.predict_proba(students)

placed_index = list(best_rf_model.classes_).index(1)

students["Placement_Probability"] = (
    probabilities[:, placed_index] * 100
).round(2)

print(students)

# =========================================================
# Save Prediction Results
# =========================================================

students.to_csv(
    "placement_prediction_results.csv",
    index=False
)

print("\nPrediction results saved successfully.")

# =========================================================
# Final Project Summary
# =========================================================

print("\n" + "=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)

print(f"Dataset Size               : {len(df)} Students")
print(f"Training Samples           : {len(X_train)}")
print(f"Testing Samples            : {len(X_test)}")

print("\nBaseline Model Accuracy")
print("------------------------------")
print(f"KNN               : {knn_accuracy:.4f}")
print(f"Random Forest     : {rf_accuracy:.4f}")

print("\nTuned Model Accuracy")
print("------------------------------")
print(f"Tuned KNN         : {knn_tuned_accuracy:.4f}")
print(f"Tuned RandomForest: {rf_tuned_accuracy:.4f}")

print("\nCross Validation")
print("------------------------------")
print(f"Average CV Score  : {average_score:.4f}")
print(f"Best CV Score     : {maximum_score:.4f}")

print("\nBest Hyperparameters")
print("------------------------------")
print(f"KNN               : {knn_grid.best_params_}")
print(f"Random Forest     : {rf_grid.best_params_}")

print("\nFeatures Used")
print("------------------------------")
for feature in X.columns:
    print(f"• {feature}")

print("\nProject Features")
print("------------------------------")
print("✓ Student Dataset Generation")
print("✓ Label Encoding")
print("✓ Train-Test Split")
print("✓ Baseline KNN Model")
print("✓ Baseline Random Forest Model")
print("✓ Cross Validation")
print("✓ Hyperparameter Tuning")
print("✓ GridSearchCV")
print("✓ Placement Prediction")
print("✓ Placement Probability")
print("✓ Multiple Student Prediction")
print("✓ CSV Export")

print("\n" + "=" * 70)
print("DAY 08 PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)