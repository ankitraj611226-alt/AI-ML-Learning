"""
====================================================================
Course      : AI & Machine Learning Internship
Day         : 11
Lecture     : 03

Topic       : Import Machine Learning Models

Author      : Ankit Raj

Description :
--------------
This program:

1. Loads the placement dataset
2. Displays dataset preview
3. Imports all Machine Learning models
4. Checks optional libraries
5. Displays available models

====================================================================
"""

# ================================================================
# Import Required Libraries
# ================================================================

import os
import pandas as pd

# ================================================================
# Get Current Folder
# ================================================================

current_folder = os.path.dirname(os.path.abspath(__file__))

# ================================================================
# Dataset Path
# ================================================================

csv_path = os.path.join(current_folder, "placement_dataset.csv")

# ================================================================
# Load Dataset
# ================================================================

try:
    df = pd.read_csv(csv_path)

    print("=" * 70)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 70)

    print(df.head())

except FileNotFoundError:

    print("=" * 70)
    print("ERROR : placement_dataset.csv NOT FOUND")
    print("=" * 70)

    print("Expected Location:")
    print(csv_path)

    exit()

# ================================================================
# Import Machine Learning Models
# ================================================================

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC

# ================================================================
# Import XGBoost
# ================================================================

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

# ================================================================
# Import LightGBM
# ================================================================

try:
    from lightgbm import LGBMClassifier
    lightgbm_available = True
except ImportError:
    lightgbm_available = False

# ================================================================
# Import CatBoost
# ================================================================

try:
    from catboost import CatBoostClassifier
    catboost_available = True
except ImportError:
    catboost_available = False

# ================================================================
# Display Imported Models
# ================================================================

print("\n" + "=" * 70)
print("MACHINE LEARNING MODELS")
print("=" * 70)

models = [

    "Logistic Regression",

    "K-Nearest Neighbors",

    "Decision Tree",

    "Random Forest",

    "Extra Trees",

    "AdaBoost",

    "Gradient Boosting",

    "Support Vector Machine"

]

if xgb_available:
    models.append("XGBoost")

if lightgbm_available:
    models.append("LightGBM")

if catboost_available:
    models.append("CatBoost")

for index, model in enumerate(models, start=1):
    print(f"{index}. {model}")

print("\n" + "=" * 70)
print("TOTAL MODELS :", len(models))
print("=" * 70)

print("\nLecture 03 Completed Successfully.")