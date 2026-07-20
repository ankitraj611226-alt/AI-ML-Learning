"""
====================================================================
Course      : AI & Machine Learning Internship
Day         : 11
Lecture     : 04

Topic       : Create Models Dictionary

Author      : Ankit Raj

Description :
--------------
This program:

1. Loads the dataset
2. Creates Machine Learning model objects
3. Stores all models in a dictionary

====================================================================
"""

# ================================================================
# Import Required Libraries
# ================================================================

import os
import pandas as pd

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

# ================================================================
# Load Dataset
# ================================================================

current_folder = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(current_folder, "placement_dataset.csv")

df = pd.read_csv(csv_path)

print("=" * 70)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 70)

print(df.head())

# ================================================================
# Create Model Dictionary
# ================================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "K-Nearest Neighbors":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "Extra Trees":
        ExtraTreesClassifier(
            n_estimators=100,
            random_state=42
        ),

    "AdaBoost":
        AdaBoostClassifier(
            n_estimators=100,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        ),

    "Support Vector Machine":
        SVC(
            probability=True,
            random_state=42
        ),

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

# ================================================================
# Display Models
# ================================================================

print("\n" + "=" * 70)
print("MODEL OBJECTS CREATED")
print("=" * 70)

for index, model_name in enumerate(models.keys(), start=1):
    print(f"{index}. {model_name}")

print("\n" + "=" * 70)
print("TOTAL MODELS :", len(models))
print("=" * 70)

print("\nLecture 04 Completed Successfully.")