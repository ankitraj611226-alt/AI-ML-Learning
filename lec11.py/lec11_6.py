"""
====================================================================
Course      : AI & Machine Learning Internship
Day         : 11
Lecture     : 05

Topic       : Training Multiple Machine Learning Models

Author      : Ankit Raj

Description :
--------------
This program:

1. Loads dataset
2. Prepares X and y
3. Splits data
4. Creates all model objects
5. Trains every model automatically
6. Measures training time

====================================================================
"""

# ==========================================================
# Import Libraries
# ==========================================================

import os
import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)

from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ==========================================================
# Load Dataset
# ==========================================================

current_folder = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(current_folder, "placement_dataset.csv")

df = pd.read_csv(csv_path)

print("=" * 70)
print("DATASET LOADED")
print("=" * 70)

# ==========================================================
# Encode Target Column
# ==========================================================

encoder = LabelEncoder()

df["Placement_Status"] = encoder.fit_transform(
    df["Placement_Status"]
)

# ==========================================================
# Features and Target
# ==========================================================

X = df.drop("Placement_Status", axis=1)

y = df["Placement_Status"]

# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================================
# Create Model Dictionary
# ==========================================================

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000, random_state=42),

    "KNN":
        KNeighborsClassifier(),

    "Decision Tree":
        DecisionTreeClassifier(max_depth=5, random_state=42),

    "Random Forest":
        RandomForestClassifier(n_estimators=100, random_state=42),

    "Extra Trees":
        ExtraTreesClassifier(n_estimators=100, random_state=42),

    "AdaBoost":
        AdaBoostClassifier(n_estimators=100, random_state=42),

    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        ),

    "SVM":
        SVC(probability=True, random_state=42),

    "XGBoost":
        XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        ),

    "LightGBM":
        LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        ),

    "CatBoost":
        CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            random_seed=42,
            verbose=0
        )

}

# ==========================================================
# Train Models
# ==========================================================

print("\n")
print("=" * 70)
print("TRAINING STARTED")
print("=" * 70)

results = []

for model_name, model in models.items():

    print(f"\nTraining : {model_name}")

    start_time = time.time()

    model.fit(X_train, y_train)

    end_time = time.time()

    training_time = end_time - start_time

    results.append({
        "Model": model_name,
        "Training Time": round(training_time, 4)
    })

    print("Training Completed")

    print(f"Time : {training_time:.4f} Seconds")

print("\n")
print("=" * 70)
print("ALL MODELS TRAINED SUCCESSFULLY")
print("=" * 70)

# ==========================================================
# Display Training Time
# ==========================================================

results_df = pd.DataFrame(results)

print(results_df)