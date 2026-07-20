"""
==============================================
Lecture 05

Topic:
Overfitting
Underfitting
==============================================
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# ==========================================
# Create Dataset
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

# ==========================================
# Features
# ==========================================

X = df.drop("Placement_Status", axis=1)

y = df["Placement_Status"]

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)

# ==========================================
# Train Model
# ==========================================

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# ==========================================
# Predictions
# ==========================================

train_prediction = model.predict(X_train)

test_prediction = model.predict(X_test)

# ==========================================
# Accuracy
# ==========================================

train_accuracy = accuracy_score(
    y_train,
    train_prediction
)

test_accuracy = accuracy_score(
    y_test,
    test_prediction
)

print("Training Accuracy :", round(train_accuracy,4))

print("Testing Accuracy  :", round(test_accuracy,4))