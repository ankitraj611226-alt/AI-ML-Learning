"""
====================================================
Lecture 10

Topic:
Machine Learning Pipeline
====================================================
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# =====================================
# Dataset
# =====================================

np.random.seed(42)

student_data = {

    "Attendance": np.random.randint(60,101,500),

    "Coding_Score": np.random.randint(40,101,500),

    "Python_Score": np.random.randint(40,101,500),

    "DSA_Score": np.random.randint(35,101,500),

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

X = df.drop("Placement_Status",axis=1)

y = df["Placement_Status"]

X_train,X_test,y_train,y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)

# =====================================
# Create Pipeline
# =====================================

pipeline = Pipeline([

    ("scaler", StandardScaler()),

    ("feature_selection", SelectKBest(
        score_func=f_classif,
        k=4
    )),

    ("model", RandomForestClassifier(
        random_state=42
    ))

])

# =====================================
# Train Pipeline
# =====================================

pipeline.fit(
    X_train,
    y_train
)

# =====================================
# Prediction
# =====================================

prediction = pipeline.predict(X_test)

accuracy = accuracy_score(
    y_test,
    prediction
)

print("Accuracy")

print(round(accuracy,4))