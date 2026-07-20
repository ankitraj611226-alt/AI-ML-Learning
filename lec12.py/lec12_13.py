"""
======================================================
Lecture 14

Topic:
ROC Curve
Precision
Recall
F1 Score
======================================================
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (

    confusion_matrix,

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score,

    roc_curve

)

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
# Train Model
# =====================================

model = RandomForestClassifier(
    random_state=42
)

model.fit(X_train,y_train)

prediction = model.predict(X_test)

probability = model.predict_proba(X_test)[:,1]

# =====================================
# Metrics
# =====================================

print("Accuracy :",accuracy_score(y_test,prediction))

print("Precision :",precision_score(y_test,prediction))

print("Recall :",recall_score(y_test,prediction))

print("F1 Score :",f1_score(y_test,prediction))

print("ROC AUC :",roc_auc_score(
    y_test,
    probability
))

# =====================================
# Confusion Matrix
# =====================================

cm = confusion_matrix(
    y_test,
    prediction
)

print()

print(cm)