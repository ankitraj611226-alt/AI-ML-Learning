"""
=========================================================
      Day 08 - Label Encoding & One-Hot Encoding
---------------------------------------------------------
Author      : Ankit Raj
Language    : Python
Description : Encoding categorical features for
              Machine Learning.
=========================================================
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

print("=" * 60)
print("DAY 08 - LABEL ENCODING & ONE-HOT ENCODING")
print("=" * 60)

# -------------------------------------------------------
# Create Dataset
# -------------------------------------------------------

student_data = {
    "Student_ID": [101, 102, 103, 104, 105, 106],
    "Name": ["Ankit", "Rahul", "Priya", "Aman", "Neha", "Riya"],
    "Gender": ["Male", "Male", "Female", "Male", "Female", "Female"],
    "Department": ["AI", "CSE", "AI", "IT", "CSE", "AI"],
    "Performance": [
        "Excellent",
        "Good",
        "Good",
        "Average",
        "Excellent",
        "Average"
    ]
}

df = pd.DataFrame(student_data)

print("\nOriginal Dataset")
print(df)

# -------------------------------------------------------
# Label Encoding
# -------------------------------------------------------

label_encoder = LabelEncoder()

df["Gender_Encoded"] = label_encoder.fit_transform(df["Gender"])

df["Performance_Encoded"] = label_encoder.fit_transform(df["Performance"])

print("\nDataset After Label Encoding")
print(df)

# -------------------------------------------------------
# One-Hot Encoding
# -------------------------------------------------------

department_encoded = pd.get_dummies(
    df["Department"],
    prefix="Department"
)

df = pd.concat([df, department_encoded], axis=1)

print("\nDataset After One-Hot Encoding")
print(df)

# -------------------------------------------------------
# Remove Original Department Column
# -------------------------------------------------------

df = df.drop(columns=["Department"])

print("\nFinal Dataset")
print(df)

# -------------------------------------------------------
# Save Dataset
# -------------------------------------------------------

df.to_csv("encoded_student_dataset.csv", index=False)

print("\nEncoded dataset saved successfully.")

print("\nDay 08 Completed Successfully!")