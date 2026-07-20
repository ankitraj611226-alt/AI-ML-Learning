"""
=========================================================
Lecture 8 - Program 4
Topic : Before vs After Hyperparameter Tuning

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
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
# Baseline Model
# =========================================================

baseline_model = RandomForestClassifier(random_state=42)

baseline_model.fit(X_train, y_train)

baseline_prediction = baseline_model.predict(X_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_prediction
)

# =========================================================
# Hyperparameter Tuning
# =========================================================

param_grid = {
    "n_estimators":[50,100,200],
    "max_depth":[3,5,10,None],
    "min_samples_split":[2,5,10]
}

grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train,y_train)

best_model = grid.best_estimator_

tuned_prediction = best_model.predict(X_test)

tuned_accuracy = accuracy_score(
    y_test,
    tuned_prediction
)

# =========================================================
# Comparison Table
# =========================================================

comparison = pd.DataFrame({

    "Model":[
        "Baseline RF",
        "Tuned RF"
    ],

    "Accuracy":[
        baseline_accuracy,
        tuned_accuracy
    ]

})

print("="*60)
print("Model Comparison")
print("="*60)

print(comparison)

# =========================================================
# Visualization
# =========================================================

plt.figure(figsize=(7,5))

plt.bar(
    comparison["Model"],
    comparison["Accuracy"]
)

plt.title("Before vs After Hyperparameter Tuning")

plt.xlabel("Model")

plt.ylabel("Accuracy")

plt.ylim(0.80,1.00)

plt.grid(axis="y")

plt.show()