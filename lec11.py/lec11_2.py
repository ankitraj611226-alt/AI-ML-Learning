"""
====================================================================
Course      : AI & Machine Learning Internship
Day         : 11
Lecture     : 02

Topic       : Dataset Preparation

Author      : Ankit Raj

Description :
--------------
This program performs all preprocessing steps
required before training Machine Learning models.

Topics Covered
--------------
1. Dataset Generation
2. Dataset Information
3. Statistical Summary
4. Missing Value Checking
5. Label Encoding
6. Feature Matrix (X)
7. Target Variable (y)
8. Train-Test Split

====================================================================
"""

# ================================================================
# Import Required Libraries
# ================================================================

import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ================================================================
# Set Random Seed
# ================================================================

random.seed(42)
np.random.seed(42)

# ================================================================
# Create Empty Dictionary
# ================================================================

student_data = {
    "Attendance": [],
    "Coding_Score": [],
    "DSA_Score": [],
    "Aptitude": [],
    "Communication": [],
    "Projects": [],
    "Internships": [],
    "CGPA": [],
    "Placement_Status": []
}

# ================================================================
# Generate Synthetic Dataset
# ================================================================

TOTAL_STUDENTS = 5000

for _ in range(TOTAL_STUDENTS):

    attendance = random.randint(40, 100)
    coding = random.randint(20, 100)
    dsa = random.randint(20, 100)
    aptitude = random.randint(20, 100)
    communication = random.randint(20, 100)

    projects = random.randint(0, 5)
    internships = random.randint(0, 3)

    cgpa = round(random.uniform(5.0, 10.0), 2)

    score = (
        coding * 0.22 +
        dsa * 0.18 +
        aptitude * 0.15 +
        communication * 0.15 +
        attendance * 0.08 +
        cgpa * 6 +
        projects * 4 +
        internships * 5
    )

    placement = "Placed" if score >= 70 else "Not Placed"

    student_data["Attendance"].append(attendance)
    student_data["Coding_Score"].append(coding)
    student_data["DSA_Score"].append(dsa)
    student_data["Aptitude"].append(aptitude)
    student_data["Communication"].append(communication)
    student_data["Projects"].append(projects)
    student_data["Internships"].append(internships)
    student_data["CGPA"].append(cgpa)
    student_data["Placement_Status"].append(placement)

# ================================================================
# Create DataFrame
# ================================================================

df = pd.DataFrame(student_data)

# ================================================================
# Display Dataset Information
# ================================================================

print("=" * 70)
print("FIRST FIVE RECORDS")
print("=" * 70)
print(df.head())

print("\n")

print("=" * 70)
print("DATASET SHAPE")
print("=" * 70)
print(df.shape)

print("\n")

print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)
df.info()

print("\n")

print("=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)
print(df.describe())

print("\n")

print("=" * 70)
print("CHECK MISSING VALUES")
print("=" * 70)
print(df.isnull().sum())

print("\n")

print("=" * 70)
print("PLACEMENT DISTRIBUTION")
print("=" * 70)
print(df["Placement_Status"].value_counts())

# ================================================================
# Encode Target Variable
# ================================================================

encoder = LabelEncoder()

df["Placement_Status"] = encoder.fit_transform(
    df["Placement_Status"]
)

print("\n")

print("=" * 70)
print("ENCODED TARGET")
print("=" * 70)
print(df["Placement_Status"].head())

# ================================================================
# Separate Features and Target
# ================================================================

X = df.drop("Placement_Status", axis=1)

y = df["Placement_Status"]

# ================================================================
# Train Test Split
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ================================================================
# Dataset Ready
# ================================================================

print("\n")

print("=" * 70)
print("DATASET READY")
print("=" * 70)

print("Training Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])

print("\n")

print("Feature Columns")
print("----------------")
print(list(X.columns))

print("\n")

print("Target Column")
print("-------------")
print("Placement_Status")

print("\nDataset is Ready for Machine Learning.")