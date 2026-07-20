"""
====================================================================
Course      : AI & Machine Learning Internship
Day         : 11
Lecture     : 07

Topic       : Model Evaluation

Author      : Ankit Raj

Description :
--------------
This program:

1. Loads dataset
2. Trains all models
3. Predicts on test data
4. Evaluates every model using:

   • Accuracy
   • Precision
   • Recall
   • F1 Score
   • ROC-AUC Score
   • Balanced Accuracy

====================================================================
"""

# ==========================================================
# Import Libraries
# ==========================================================

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score
)

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
# Models Dictionary
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
# Evaluate Models
# ==========================================================

results = []

print("\n")
print("=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

for model_name, model in models.items():

    print(f"\nEvaluating {model_name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred
    )

    try:
        probability = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(
            y_test,
            probability
        )

    except:

        roc_auc = "Not Available"

    results.append({

        "Model": model_name,

        "Accuracy": round(accuracy, 4),

        "Precision": round(precision, 4),

        "Recall": round(recall, 4),

        "F1 Score": round(f1, 4),

        "ROC AUC": roc_auc if isinstance(
            roc_auc,
            str
        ) else round(roc_auc, 4),

        "Balanced Accuracy":
            round(balanced_accuracy, 4)

    })

# ==========================================================
# Create Result DataFrame
# ==========================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(results_df)