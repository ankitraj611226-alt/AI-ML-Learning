"""
=========================================================
            Day 07 - Feature Engineering
---------------------------------------------------------
Author      : Ankit Raj
Language    : Python
Description : Feature Engineering using Pandas
=========================================================
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("DAY 07 - FEATURE ENGINEERING")
print("=" * 60)

# -------------------------------------------------------
# Create Dataset
# -------------------------------------------------------

student_data = {
    "Student_ID": [101, 102, 103, 104, 105, 106],
    "Name": ["Ankit", "Rahul", "Priya", "Aman", "Neha", "Riya"],
    "Study_Hours": [4, 6, 5, 3, 7, 8],
    "Attendance": [85, 92, 88, 75, 97, 95],
    "Marks": [82, 91, 76, 69, 95, 98]
}

df = pd.DataFrame(student_data)

print("\nOriginal Dataset")
print(df)

# -------------------------------------------------------
# Create New Feature
# -------------------------------------------------------

df["Total_Score"] = df["Marks"] + df["Attendance"]

print("\nDataset with Total Score")
print(df)

# -------------------------------------------------------
# Average Score Feature
# -------------------------------------------------------

df["Average_Score"] = (
    df["Marks"] + df["Attendance"]
) / 2

print("\nAverage Score")
print(df)

# -------------------------------------------------------
# Performance Category
# -------------------------------------------------------

df["Performance"] = np.where(
    df["Marks"] >= 90,
    "Excellent",
    np.where(
        df["Marks"] >= 75,
        "Good",
        "Needs Improvement"
    )
)

print("\nPerformance Category")
print(df)

# -------------------------------------------------------
# Efficiency Feature
# -------------------------------------------------------

df["Efficiency"] = (
    df["Marks"] / df["Study_Hours"]
).round(2)

print("\nLearning Efficiency")
print(df)

# -------------------------------------------------------
# Rank Students
# -------------------------------------------------------

df["Rank"] = (
    df["Marks"]
    .rank(ascending=False, method="dense")
    .astype(int)
)

print("\nStudent Ranking")
print(df)

# -------------------------------------------------------
# Sort by Rank
# -------------------------------------------------------

df = df.sort_values("Rank")

print("\nFinal Dataset")
print(df)

# -------------------------------------------------------
# Save Dataset
# -------------------------------------------------------

df.to_csv("feature_engineered_dataset.csv", index=False)

print("\nFeature engineered dataset saved successfully.")

print("\nDay 07 Completed Successfully!")