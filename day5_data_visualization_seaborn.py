"""
=========================================================
      Day 06 - Data Visualization using Seaborn
---------------------------------------------------------
Author      : Ankit Raj
Language    : Python
Description : Learn Data Visualization using Seaborn.
=========================================================
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("=" * 60)
print("DAY 06 - DATA VISUALIZATION USING SEABORN")
print("=" * 60)

# -------------------------------------------------------
# Create Dataset
# -------------------------------------------------------

student_data = {
    "Name": ["Ankit", "Rahul", "Priya", "Aman", "Neha", "Riya", "Karan", "Sneha"],
    "Study_Hours": [4, 6, 5, 3, 7, 8, 2, 6],
    "Marks": [82, 91, 76, 69, 95, 98, 65, 89],
    "Attendance": [85, 92, 88, 75, 97, 95, 70, 90]
}

df = pd.DataFrame(student_data)

print("\nDataset")
print(df)

# -------------------------------------------------------
# Theme
# -------------------------------------------------------

sns.set_theme(style="whitegrid")

# -------------------------------------------------------
# Scatter Plot
# -------------------------------------------------------

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="Study_Hours", y="Marks", hue="Attendance", s=100)

plt.title("Study Hours vs Marks")
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# Bar Plot
# -------------------------------------------------------

plt.figure(figsize=(8,5))
sns.barplot(data=df, x="Name", y="Marks")

plt.title("Student Marks")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# Histogram
# -------------------------------------------------------

plt.figure(figsize=(8,5))
sns.histplot(df["Marks"], bins=5, kde=True)

plt.title("Marks Distribution")
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# Box Plot
# -------------------------------------------------------

plt.figure(figsize=(6,5))
sns.boxplot(y=df["Marks"])

plt.title("Marks Box Plot")
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# Correlation Heatmap
# -------------------------------------------------------

plt.figure(figsize=(6,5))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="Blues"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

print("\nDay 06 Completed Successfully!")