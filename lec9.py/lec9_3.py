"""
=========================================================
Lecture 9 - Program 3
Topic : Before vs After Feature Selection

Author : Ankit Raj
Course : AI & ML Internship
=========================================================
"""

# =========================================================
# Import Libraries
# =========================================================

import random
import pandas as pd
import matplotlib.pyplot as plt

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
# Train Baseline Model
# =========================================================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    random_state=42
)

model.fit(X_train, y_train)

# =========================================================
# Baseline Accuracy
# =========================================================

baseline_prediction = model.predict(X_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_prediction
)

# =========================================================
# Feature Importance
# =========================================================

importance_df = pd.DataFrame({

    "Feature": X.columns,
    "Importance": model.feature_importances_

})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# =========================================================
# Select Important Features
# =========================================================

selected_features = importance_df[
    importance_df["Importance"] > 0.15
]

print("=" * 60)
print("Selected Features")
print("=" * 60)

print(selected_features)

# =========================================================
# New Dataset
# =========================================================

X_selected = df[
    selected_features["Feature"]
]

# =========================================================
# Train Test Split
# =========================================================

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_selected,
    y,
    test_size=0.20,
    random_state=42
)

# =========================================================
# Train Model with Selected Features
# =========================================================

selected_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    random_state=42
)

selected_model.fit(
    X_train_s,
    y_train_s
)

# =========================================================
# Selected Model Accuracy
# =========================================================

selected_prediction = selected_model.predict(
    X_test_s
)

selected_accuracy = accuracy_score(
    y_test_s,
    selected_prediction
)

# =========================================================
# Comparison Table
# =========================================================

comparison = pd.DataFrame({

    "Model": [
        "All Features",
        "Selected Features"
    ],

    "Accuracy": [
        baseline_accuracy,
        selected_accuracy
    ]

})

print("\n" + "=" * 60)
print("Accuracy Comparison")
print("=" * 60)

print(comparison)

# =========================================================
# Visualization
# =========================================================

plt.figure(figsize=(6, 4))

plt.bar(
    comparison["Model"],
    comparison["Accuracy"]
)

plt.title("Before vs After Feature Selection")

plt.xlabel("Model")

plt.ylabel("Accuracy")

plt.ylim(0.70, 1.00)

plt.grid(axis="y")

plt.tight_layout()

plt.show()