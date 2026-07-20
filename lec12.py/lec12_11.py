"""
=====================================================
Lecture 12

Topic:
Explainable AI (Feature Importance)
=====================================================
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

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

    df["Coding_Score"] * 0.4 +

    df["Python_Score"] * 0.3 +

    df["CGPA"] * 5

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

# =====================================
# Train Model
# =====================================

model = RandomForestClassifier(
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# =====================================
# Feature Importance
# =====================================

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print(importance)