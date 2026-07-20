"""
====================================================================
Course      : AI & Machine Learning Internship
Day         : 11
Lecture     : 01

Topic       : Synthetic Dataset Generation

Author      : Ankit Raj

Description :
--------------
This program generates a synthetic placement dataset
containing 5000 student records.

Each student has:

1. Attendance
2. Coding Score
3. DSA Score
4. Aptitude
5. Communication Skills
6. Number of Projects
7. Number of Internships
8. CGPA
9. Placement Status

This dataset will be used throughout Day 11
for comparing multiple Machine Learning models.

====================================================================
"""

# ================================================================
# Import Required Libraries
# ================================================================

import random
import numpy as np
import pandas as pd

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
# Generate Synthetic Student Records
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

    # ------------------------------------------------------------
    # Placement Score Formula
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Placement Decision
    # ------------------------------------------------------------

    if score >= 70:
        placement = "Placed"
    else:
        placement = "Not Placed"

    # ------------------------------------------------------------
    # Store Student Data
    # ------------------------------------------------------------

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
# Convert Dictionary into DataFrame
# ================================================================

df = pd.DataFrame(student_data)

# ================================================================
# Display First Five Records
# ================================================================

print("=" * 70)
print("FIRST FIVE RECORDS")
print("=" * 70)

print(df.head())

# ================================================================
# Display Dataset Shape
# ================================================================

print("\n" + "=" * 70)
print("DATASET SHAPE")
print("=" * 70)

print(df.shape)

# ================================================================
# Placement Distribution
# ================================================================

print("\n" + "=" * 70)
print("PLACEMENT DISTRIBUTION")
print("=" * 70)

print(df["Placement_Status"].value_counts())

# ================================================================
# Dataset Successfully Generated
# ================================================================

print("\n" + "=" * 70)
print("DATASET GENERATED SUCCESSFULLY")
print("=" * 70)

print(f"Total Students : {TOTAL_STUDENTS}")
# ================================================================
# Create DataFrame
# ================================================================

df = pd.DataFrame(student_data)

# ================================================================
# Save DataFrame as CSV
# ================================================================

df.to_csv("placement_dataset.csv", index=False)

print("=" * 60)
print("CSV FILE CREATED SUCCESSFULLY")
print("=" * 60)

print("File Name : placement_dataset.csv")