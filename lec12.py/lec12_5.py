"""
===============================================
Lecture 06

Topic:
GridSearchCV
===============================================
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import (

    train_test_split,

    GridSearchCV

)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# ==========================================
# Dataset
# ==========================================

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

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)

# ==========================================
# Parameter Grid
# ==========================================

parameter_grid = {

    "n_estimators":[100,200,300],

    "max_depth":[5,10,15],

    "min_samples_split":[2,5],

    "min_samples_leaf":[1,2]

}

# ==========================================
# Grid Search
# ==========================================

grid_search = GridSearchCV(

    estimator=RandomForestClassifier(
        random_state=42
    ),

    param_grid=parameter_grid,

    cv=5,

    scoring="accuracy",

    n_jobs=-1

)

grid_search.fit(

    X_train,

    y_train

)

# ==========================================
# Best Parameters
# ==========================================

print("Best Parameters")

print(grid_search.best_params_)

print()

print("Best Cross Validation Score")

print(round(grid_search.best_score_,4))

# ==========================================
# Best Model
# ==========================================

best_model = grid_search.best_estimator_

prediction = best_model.predict(X_test)

accuracy = accuracy_score(

    y_test,

    prediction

)

print()

print("Testing Accuracy")

print(round(accuracy,4))