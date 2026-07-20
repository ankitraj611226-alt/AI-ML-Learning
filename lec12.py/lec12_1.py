"""
=========================================================
Lecture 02
Topic : Cross Validation
=========================================================
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

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

X = df.drop(
    "Placement_Status",
    axis=1
)

y = df["Placement_Status"]

# ============================================
# Train Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)

# ============================================
# Create Model
# ============================================

model = LogisticRegression(max_iter=1000)

# ============================================
# Cross Validation
# ============================================

scores = cross_val_score(

    estimator=model,

    X=X,

    y=y,

    cv=5,

    scoring="accuracy"

)

print("Cross Validation Scores")

print(scores)

print()

print("Average Accuracy")

print(scores.mean())

print()

print("Standard Deviation")

print(scores.std())