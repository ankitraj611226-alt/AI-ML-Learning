"""
======================================================
Lecture 09

Topic:
Feature Selection
======================================================
"""

import numpy as np
import pandas as pd

from sklearn.feature_selection import (

    SelectKBest,

    f_classif

)

# ==========================================
# Dataset
# ==========================================

np.random.seed(42)

student_data = {

    "Attendance": np.random.randint(60,101,100),

    "Coding_Score": np.random.randint(40,101,100),

    "DSA_Score": np.random.randint(35,101,100),

    "Python_Score": np.random.randint(40,101,100),

    "SQL_Score": np.random.randint(35,101,100),

    "Projects": np.random.randint(0,6,100),

    "Resume_Score": np.random.randint(50,101,100),

    "Communication_Skills": np.random.randint(5,11,100),

    "CGPA": np.round(
        np.random.uniform(5.5,9.8,100),
        2
    )

}

df = pd.DataFrame(student_data)

placement_score = (

    df["Coding_Score"] * 0.4 +

    df["Python_Score"] * 0.3 +

    df["CGPA"] * 5

)

df["Placement_Status"] = np.where(

    placement_score >= 80,

    1,

    0

)

# ==========================================
# Features and Target
# ==========================================

X = df.drop(

    "Placement_Status",

    axis=1

)

y = df["Placement_Status"]

# ==========================================
# Feature Selection
# ==========================================

selector = SelectKBest(

    score_func=f_classif,

    k=5

)

X_selected = selector.fit_transform(

    X,

    y

)

selected_features = X.columns[

    selector.get_support()

]

print("Selected Features:\n")

print(selected_features)

print()

print("Feature Scores:\n")

feature_scores = pd.DataFrame({

    "Feature": X.columns,

    "Score": selector.scores_

})

print(

    feature_scores.sort_values(

        by="Score",

        ascending=False

    )

)