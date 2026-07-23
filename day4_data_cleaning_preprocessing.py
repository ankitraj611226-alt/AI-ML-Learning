"""
=========================================================
        Day 04 - Data Cleaning & Preprocessing
---------------------------------------------------------
Author      : Ankit Raj
Language    : Python
Description : Handling Missing Values, Duplicates,
              Data Cleaning and Feature Engineering
=========================================================
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("DAY 04 - DATA CLEANING & PREPROCESSING")
print("=" * 60)

# -------------------------------------------------------
# Create Dataset
# -------------------------------------------------------

student_data = {
    "Student_ID": [101, 102, 103, 104, 105, 105],
    "Name": ["Ankit", "Rahul", "Priya", "Aman", "Neha", "Neha"],
    "Age": [20, 21, np.nan, 22, 20, 20],
    "Study_Hours": [4, 6, 5, np.nan, 7, 7],
    "Attendance": [85, 92, 88, 75, np.nan, np.nan],
    "Marks": [82, 91, np.nan, 69, 95, 95]
}

df = pd.DataFrame(student_data)

print("\nOriginal Dataset")
print(df)

# -------------------------------------------------------
# Missing Values
# -------------------------------------------------------

print("\nMissing Values")
print(df.isnull().sum())

# -------------------------------------------------------
# Fill Missing Values
# -------------------------------------------------------

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Study_Hours"] = df["Study_Hours"].fillna(df["Study_Hours"].mean())
df["Attendance"] = df["Attendance"].fillna(df["Attendance"].mean())
df["Marks"] = df["Marks"].fillna(df["Marks"].mean())

print("\nDataset After Filling Missing Values")
print(df)

# -------------------------------------------------------
# Duplicate Records
# -------------------------------------------------------

print("\nDuplicate Rows:", df.duplicated().sum())

df = df.drop_duplicates()

print("\nDataset After Removing Duplicates")
print(df)

# -------------------------------------------------------
# Data Types
# -------------------------------------------------------

print("\nData Types")
print(df.dtypes)

# -------------------------------------------------------
# Statistical Summary
# -------------------------------------------------------

print("\nStatistical Summary")
print(df.describe())

# -------------------------------------------------------
# Add New Feature
# -------------------------------------------------------

df["Performance"] = np.where(df["Marks"] >= 85, "Excellent", "Needs Improvement")

print("\nDataset with Performance Column")
print(df)

# -------------------------------------------------------
# Sort Dataset
# -------------------------------------------------------

df = df.sort_values(by="Marks", ascending=False)

print("\nSorted Dataset")
print(df)

# -------------------------------------------------------
# Save Clean Dataset
# -------------------------------------------------------

df.to_csv("clean_student_dataset.csv", index=False)

print("\nClean dataset saved successfully.")

print("\nDay 04 Completed Successfully!")