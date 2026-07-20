df = pd.read_csv("placement_dataset.csv")

print("=" * 70)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 70)

print(df.head())

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

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

# ================================================================
# Display Imported Models
# ================================================================

print("\n" + "=" * 70)
print("IMPORTED MACHINE LEARNING MODELS")
print("=" * 70)

print("1. Logistic Regression")
print("2. K-Nearest Neighbors")
print("3. Decision Tree")
print("4. Random Forest")
print("5. Extra Trees")
print("6. AdaBoost")
print("7. Gradient Boosting")
print("8. Support Vector Machine")
print("9. XGBoost")
print("10. LightGBM")
print("11. CatBoost")

print("\n")

print("=" * 70)
print("TOTAL MODELS IMPORTED :", 11)
print("=" * 70)

print("\nReady to Create Model Objects...")