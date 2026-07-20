"""
=========================================================
Lecture 9 - Program 1
Topic : Feature Importance

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

print("=" * 60)
print("First Five Records")
print("=" * 60)

print(df.head())

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

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    random_state=42
)

model.fit(X_train, y_train)

# =========================================================
# Calculate Feature Importance
# =========================================================

importance = model.feature_importances_

print("\n" + "=" * 60)
print("Feature Importance Scores")
print("=" * 60)

print(importance)

# =========================================================
# Create Feature Importance Table
# =========================================================

importance_df = pd.DataFrame({

    "Feature": X.columns,
    "Importance": importance

})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("Feature Importance Table")
print("=" * 60)

print(importance_df)

# =========================================================
# Most Important Feature
# =========================================================

print("\nMost Important Feature")

print(importance_df.iloc[0])

# =========================================================
# Least Important Feature
# =========================================================

print("\nLeast Important Feature")

print(importance_df.iloc[-1])

# =========================================================
# Visualization
# =========================================================

plt.figure(figsize=(8, 5))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance")

plt.tight_layout()

plt.show()