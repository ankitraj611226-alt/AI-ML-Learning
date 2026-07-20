"""
==================================================
Lecture 04

Topic:
Compare Multiple Models using Cross Validation
==================================================
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

# =======================================
# Dataset
# =======================================

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

X = df.drop("Placement_Status", axis=1)

y = df["Placement_Status"]

# =======================================
# Models Dictionary
# =======================================

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(random_state=42),

    "Gradient Boosting":
        GradientBoostingClassifier(random_state=42),

    "KNN":
        KNeighborsClassifier(),

    "SVM":
        SVC()

}

# =======================================
# Cross Validation
# =======================================

results = []

for model_name, model in models.items():

    scores = cross_val_score(

        estimator=model,

        X=X,

        y=y,

        cv=5,

        scoring="accuracy"

    )

    results.append([

        model_name,

        round(scores.mean(),4),

        round(scores.std(),4)

    ])

# =======================================
# Result Table
# =======================================

comparison = pd.DataFrame(

    results,

    columns=[

        "Model",

        "Average Accuracy",

        "Standard Deviation"

    ]

)

comparison = comparison.sort_values(

    by="Average Accuracy",

    ascending=False

)

comparison.reset_index(

    drop=True,

    inplace=True

)

comparison.index = comparison.index + 1

print(comparison)