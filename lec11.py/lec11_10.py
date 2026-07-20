"""
====================================================================
Course      : AI & Machine Learning Internship
Day         : 11
Lecture     : 10

Topic       : Select and Save Best Machine Learning Model

Author      : Ankit Raj

Description :
--------------
This program:

1. Loads dataset
2. Trains all models
3. Evaluates every model
4. Finds the best model based on Accuracy
5. Saves the best model
6. Loads the saved model
7. Makes predictions using the saved model

====================================================================
"""

# ==========================================================
# Import Libraries
# ==========================================================

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

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

csv_path = os.path.join(
    current_folder,
    "placement_dataset.csv"
)

df = pd.read_csv(csv_path)

# ==========================================================
# Encode Target
# ==========================================================

encoder = LabelEncoder()

df["Placement_Status"] = encoder.fit_transform(
    df["Placement_Status"]
)

# ==========================================================
# Features & Target
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
# Model Dictionary
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
            verbosity=0,
            random_state=42
        ),

    "LightGBM":
        LGBMClassifier(
            verbose=-1,
            random_state=42
        ),

    "CatBoost":
        CatBoostClassifier(
            verbose=0,
            random_seed=42
        )
}

# ==========================================================
# Find Best Model
# ==========================================================

best_model = None

best_model_name = ""

best_accuracy = 0

for model_name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print(f"{model_name} : {accuracy:.4f}")

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

        best_model_name = model_name

# ==========================================================
# Display Best Model
# ==========================================================

print("\n" + "=" * 70)

print("BEST MODEL")

print("=" * 70)

print("Model Name :", best_model_name)

print("Accuracy   :", round(best_accuracy, 4))

# ==========================================================
# Save Best Model
# ==========================================================

model_path = os.path.join(
    current_folder,
    "best_model.pkl"
)

joblib.dump(
    best_model,
    model_path
)

print("\nBest Model Saved Successfully")

print("File :", model_path)

# ==========================================================
# Load Saved Model
# ==========================================================

loaded_model = joblib.load(
    model_path
)

print("\nSaved Model Loaded Successfully")

# ==========================================================
# Predict Using Saved Model
# ==========================================================

prediction = loaded_model.predict(
    X_test
)

print("\nFirst 10 Predictions")

print(prediction[:10])