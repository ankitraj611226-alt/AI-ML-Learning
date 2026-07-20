"""
=====================================================
Lecture 13

Topic:
AI Recommendation Engine
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
# Select One Student
# =====================================

student = X_test.iloc[[0]]

prediction = model.predict(student)[0]

probability = model.predict_proba(student)[0]

confidence = max(probability)

print("Prediction :", prediction)

print("Confidence :", round(confidence*100,2), "%")

# =====================================
# Recommendation Engine
# =====================================

if prediction == 1:

    print("\nRecommendation")

    print("✔ Continue Interview Preparation")

    print("✔ Practice Aptitude")

    print("✔ Improve Communication Skills")

    print("✔ Apply for Product Companies")

else:

    print("\nRecommendation")

    print("✔ Improve Coding Skills")

    print("✔ Practice DSA Daily")

    print("✔ Build More Projects")

    print("✔ Improve CGPA")