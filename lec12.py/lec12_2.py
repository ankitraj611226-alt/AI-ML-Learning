"""
=========================================================
Lecture 03

Topic :
K-Fold Cross Validation
Stratified K-Fold
=========================================================
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score
)

from sklearn.linear_model import LogisticRegression

# ============================================
# Create Dataset
# ============================================

np.random.seed(42)

student_data = {

    "Attendance": np.random.randint(60,101,500),

    "Coding_Score": np.random.randint(40,101,500),

    "DSA_Score": np.random.randint(35,101,500),

    "Python_Score": np.random.randint(40,101,500),

    "CGPA": np.round(
        np.random.uniform(5.5,9.8,500),
        2
    )

}

df = pd.DataFrame(student_data)

placement_score = (

    df["Coding_Score"]*0.4 +

    df["Python_Score"]*0.3 +

    df["CGPA"]*5

)

df["Placement_Status"] = np.where(
    placement_score >= 80,
    1,
    0
)

# ============================================
# Features and Target
# ============================================

X = df.drop("Placement_Status", axis=1)

y = df["Placement_Status"]

# ============================================
# Create Model
# ============================================

model = LogisticRegression(max_iter=1000)

# ============================================
# K-Fold Cross Validation
# ============================================

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

kfold_scores = cross_val_score(
    model,
    X,
    y,
    cv=kfold,
    scoring="accuracy"
)

print("="*50)
print("K-Fold Results")
print("="*50)

print(kfold_scores)

print("Average :", kfold_scores.mean())

# ============================================
# Stratified K-Fold
# ============================================

stratified = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

stratified_scores = cross_val_score(
    model,
    X,
    y,
    cv=stratified,
    scoring="accuracy"
)

print()

print("="*50)
print("Stratified K-Fold Results")
print("="*50)

print(stratified_scores)

print("Average :", stratified_scores.mean())