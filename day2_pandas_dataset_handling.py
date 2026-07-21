"""
=========================================================
            Day 03 - Pandas Dataset Handling
---------------------------------------------------------
Author      : Ankit Raj
Language    : Python
Description : Introduction to Pandas and DataFrame
              operations.
=========================================================
"""

import pandas as pd

print("=" * 60)
print("        DAY 03 - PANDAS DATASET HANDLING")
print("=" * 60)

# -------------------------------------------------------
# Create Dataset
# -------------------------------------------------------

student_data = {
    "Student_ID": [101, 102, 103, 104, 105],
    "Name": ["Ankit", "Rahul", "Priya", "Aman", "Neha"],
    "Age": [20, 21, 19, 22, 20],
    "Course": ["AI & ML", "CSE", "ECE", "IT", "AI & DS"],
    "Study_Hours": [4, 6, 5, 3, 7],
    "Marks": [82, 91, 76, 69, 95]
}

# -------------------------------------------------------
# Convert Dictionary into DataFrame
# -------------------------------------------------------

df = pd.DataFrame(student_data)

print("\nOriginal Dataset")
print(df)

# -------------------------------------------------------
# Head
# -------------------------------------------------------

print("\nFirst 5 Rows")
print(df.head())

# -------------------------------------------------------
# Tail
# -------------------------------------------------------

print("\nLast 5 Rows")
print(df.tail())

# -------------------------------------------------------
# Shape
# -------------------------------------------------------

print("\nDataset Shape")
print(df.shape)

# -------------------------------------------------------
# Columns
# -------------------------------------------------------

print("\nColumn Names")
print(df.columns)

# -------------------------------------------------------
# Dataset Information
# -------------------------------------------------------

print("\nDataset Information")
print(df.info())

# -------------------------------------------------------
# Statistical Summary
# -------------------------------------------------------

print("\nStatistical Summary")
print(df.describe())

# -------------------------------------------------------
# Select Single Column
# -------------------------------------------------------

print("\nMarks Column")
print(df["Marks"])

# -------------------------------------------------------
# Select Multiple Columns
# -------------------------------------------------------

print("\nName and Marks")
print(df[["Name", "Marks"]])

# -------------------------------------------------------
# Highest Marks
# -------------------------------------------------------

highest_marks = df["Marks"].max()

print("\nHighest Marks")
print(highest_marks)

# -------------------------------------------------------
# Lowest Marks
# -------------------------------------------------------

lowest_marks = df["Marks"].min()

print("\nLowest Marks")
print(lowest_marks)

# -------------------------------------------------------
# Average Marks
# -------------------------------------------------------

average_marks = df["Marks"].mean()

print("\nAverage Marks")
print(round(average_marks, 2))

# -------------------------------------------------------
# Students Scoring Above Average
# -------------------------------------------------------

print("\nStudents Scoring Above Average")

above_average = df[df["Marks"] > average_marks]

print(above_average)

# -------------------------------------------------------
# Students with Study Hours > 5
# -------------------------------------------------------

print("\nStudents Studying More Than 5 Hours")

study_hours = df[df["Study_Hours"] > 5]

print(study_hours)

# -------------------------------------------------------
# Sort Dataset by Marks
# -------------------------------------------------------

print("\nDataset Sorted by Marks")

sorted_df = df.sort_values(by="Marks", ascending=False)

print(sorted_df)

# -------------------------------------------------------
# Save Dataset
# -------------------------------------------------------

df.to_csv("student_dataset.csv", index=False)

print("\nDataset saved successfully as 'student_dataset.csv'.")

print("\nDay 03 Completed Successfully!")