# ==========================================
# Day 6 - Machine Learning using Logistic Regression
# ==========================================

# Step 1 : Import Libraries

import pandas as pd
import random

# ==========================================
# Step 2 : Create Student Dataset
# ==========================================

student_data = {
    'Attendance': [],
    'Coding_Score': [],
    'Study_Hours': [],
    'Communication_Skills': [],
    'Placement_Status': []
}

for i in range(200):

    attendance = random.randint(40, 100)
    coding = random.randint(20, 100)
    study = random.randint(1, 10)
    communication = random.randint(1, 10)

    if coding >= 60 and study >= 5 and communication >= 5:
        placement = "Placed"
    else:
        placement = "Not Placed"

    student_data['Attendance'].append(attendance)
    student_data['Coding_Score'].append(coding)
    student_data['Study_Hours'].append(study)
    student_data['Communication_Skills'].append(communication)
    student_data['Placement_Status'].append(placement)

# ==========================================
# Step 3 : Convert Dictionary into DataFrame
# ==========================================

df = pd.DataFrame(student_data)

print(df.head())

# ==========================================
# Step 4 : Encode Target Variable
# ==========================================

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df["Placement_Status"] = encoder.fit_transform(df["Placement_Status"])

# ==========================================
# Step 5 : Features (X) and Target (y)
# ==========================================

X = df.drop("Placement_Status", axis=1)

y = df["Placement_Status"]

# ==========================================
# Step 6 : Train Test Split
# ==========================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Step 7 : Import Logistic Regression
# ==========================================

from sklearn.linear_model import LogisticRegression

# ==========================================
# Step 8 : Create Model
# ==========================================

model = LogisticRegression()

# ==========================================
# Step 9 : Train Model
# ==========================================

model.fit(X_train, y_train)

# ==========================================
# Step 10 : Prediction
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# Step 11 : Accuracy
# ==========================================

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy :", accuracy)

print("Accuracy Percentage :", accuracy * 100)

# ==========================================
# Step 12 : Confusion Matrix
# ==========================================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix")

print(cm)

# ==========================================
# Step 13 : Classification Report
# ==========================================

from sklearn.metrics import classification_report

print("Classification Report")

print(classification_report(y_test, y_pred))