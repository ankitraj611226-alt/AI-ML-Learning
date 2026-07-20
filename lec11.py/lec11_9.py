"""
====================================================================
Course      : AI & Machine Learning Internship
Day         : 11
Lecture     : 09

Topic       : Compare Machine Learning Models

Author      : Ankit Raj

Description :
--------------
This program:

1. Loads model comparison results
2. Displays original results
3. Sorts models by Accuracy
4. Assigns ranking
5. Displays Top 5 models

====================================================================
"""

# ==========================================================
# Import Libraries
# ==========================================================

import os
import pandas as pd

# ==========================================================
# Get Current Folder
# ==========================================================

current_folder = os.path.dirname(os.path.abspath(__file__))

# ==========================================================
# Load Result CSV
# ==========================================================

result_path = os.path.join(
    current_folder,
    "model_comparison_results.csv"
)

results_df = pd.read_csv(result_path)

# ==========================================================
# Display Original Results
# ==========================================================

print("=" * 70)
print("ORIGINAL MODEL RESULTS")
print("=" * 70)

print(results_df)

# ==========================================================
# Sort Models by Accuracy
# ==========================================================

sorted_results = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

# ==========================================================
# Reset Index
# ==========================================================

sorted_results.reset_index(
    drop=True,
    inplace=True
)

# ==========================================================
# Create Ranking
# ==========================================================

sorted_results.insert(
    0,
    "Rank",
    range(1, len(sorted_results) + 1)
)

# ==========================================================
# Display Ranked Models
# ==========================================================

print("\n")
print("=" * 70)
print("RANKED MODEL COMPARISON")
print("=" * 70)

print(sorted_results)

# ==========================================================
# Display Top 5 Models
# ==========================================================

print("\n")
print("=" * 70)
print("TOP 5 MODELS")
print("=" * 70)

print(sorted_results.head())

# ==========================================================
# Save Ranked Results
# ==========================================================

ranked_file = os.path.join(
    current_folder,
    "ranked_model_results.csv"
)

sorted_results.to_csv(
    ranked_file,
    index=False
)

print("\n")
print("=" * 70)
print("RANKED RESULTS SAVED SUCCESSFULLY")
print("=" * 70)

print("Saved File :", ranked_file)