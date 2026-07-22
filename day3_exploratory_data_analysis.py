"""
=========================================================
            Day 04 - Exploratory Data Analysis (EDA)
---------------------------------------------------------
Author      : Ankit Raj
Language    : Python
Description : Basic Exploratory Data Analysis using Pandas
=========================================================
"""

import pandas as pd

print("=" * 60)
print("DAY 04 - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# -------------------------------------------------
# Create Dataset
# -------------------------------------------------

student_data = {
    "Student_ID": [101,102,103,104,105,106,107,108],
    "Name":["Ankit","Rahul","Priya","Aman","Neha","Riya","Karan","Sneha"],
    "Age":[20,21,19,22,20,21,23,20],
    "Study_Hours":[4,6,5,3,7,8,2,6],
    "Attendance":[85,92,88,75,97,95,70,90],
    "Marks":[82,91,76,69,95,98,65,89]
}

df = pd.DataFrame(student_data)

# -------------------------------------------------
# Display Dataset
# -------------------------------------------------

print("\nDataset")
print(df)

# -------------------------------------------------
# Dataset Information
# -------------------------------------------------

print("\nDataset Info")
df.info()

# -------------------------------------------------
# Shape
# -------------------------------------------------

print("\nShape :", df.shape)

# -------------------------------------------------
# Columns
# -------------------------------------------------

print("\nColumns")
print(df.columns)

# -------------------------------------------------
# Statistical Summary
# -------------------------------------------------

print("\nStatistical Summary")
print(df.describe())

# -------------------------------------------------
# Missing Values
# -------------------------------------------------

print("\nMissing Values")
print(df.isnull().sum())

# -------------------------------------------------
# Duplicate Rows
# -------------------------------------------------

print("\nDuplicate Rows :", df.duplicated().sum())

# -------------------------------------------------
# Average Marks
# -------------------------------------------------

print("\nAverage Marks :", df["Marks"].mean())

# -------------------------------------------------
# Highest Marks
# -------------------------------------------------

print("Highest Marks :", df["Marks"].max())

# -------------------------------------------------
# Lowest Marks
# -------------------------------------------------

print("Lowest Marks :", df["Marks"].min())

# -------------------------------------------------
# Students Above 90 Marks
# -------------------------------------------------

print("\nStudents Scoring Above 90")

print(df[df["Marks"] > 90])

# -------------------------------------------------
# Attendance Greater Than 90
# -------------------------------------------------

print("\nAttendance Above 90")

print(df[df["Attendance"] > 90])

# -------------------------------------------------
# Sort by Marks
# -------------------------------------------------

print("\nSorted by Marks")

print(df.sort_values(by="Marks", ascending=False))

# -------------------------------------------------
# Correlation
# -------------------------------------------------

print("\nCorrelation Matrix")

print(df.corr(numeric_only=True))

# -------------------------------------------------
# Save Dataset
# -------------------------------------------------

df.to_csv("student_dataset_day4.csv", index=False)

print("\nDataset saved successfully.")

print("\nDay 04 Completed Successfully!")