# ==========================================================
# AI/ML Internship - Day 7
# Multiple Machine Learning Models and Model Comparison
# ==========================================================

# ==========================================================
# Step 1 : Import Libraries
# ==========================================================

import pandas as pd
import random
import matplotlib.pyplot as plt

# ==========================================================
# Step 2 : Generate Student Dataset
# ==========================================================

student_data = {
    "Attendance": [],
    "Coding_Score": [],
    "Study_Hours": [],
    "Communication_Skills": [],
    "Placement_Status": []
}

for i in range(300):

    attendance = random.randint(40, 100)
    coding = random.randint(20, 100)
    study = random.randint(1, 10)
    communication = random.randint(1, 10)

    score = (
        coding * 0.5 +
        attendance * 0.2 +
        study * 5 +
        communication * 5
    )

    if score >= 80:
        placement = "Placed"
    else:
        placement = "Not Placed"

    student_data["Attendance"].append(attendance)
    student_data["Coding_Score"].append(coding)
    student_data["Study_Hours"].append(study)
    student_data["Communication_Skills"].append(communication)
    student_data["Placement_Status"].append(placement)

# ==========================================================
# Step 3 : Create DataFrame
# ==========================================================

df = pd.DataFrame(student_data)

print("\nFirst 5 Records\n")
print(df.head())

# ==========================================================
# Step 4 : Label Encoding
# ==========================================================

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df["Placement_Status"] = encoder.fit_transform(df["Placement_Status"])

# ==========================================================
# Step 5 : Features (X) and Target (y)
# ==========================================================

X = df[[
    "Attendance",
    "Coding_Score",
    "Study_Hours",
    "Communication_Skills"
]]

y = df["Placement_Status"]

# ==========================================================
# Step 6 : Train-Test Split
# ==========================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================================
# Step 7 : Import ML Models
# ==========================================================

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ==========================================================
# Step 8 : Create Models
# ==========================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# ==========================================================
# Step 9 : Train & Compare Models
# ==========================================================

from sklearn.metrics import accuracy_score

results = {}

for name, model in models.items():

    # Train
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy

    print(f"{name} : {accuracy:.4f}")

# ==========================================================
# Step 10 : Comparison Table
# ==========================================================

results_df = pd.DataFrame(
    list(results.items()),
    columns=["Model", "Accuracy"]
)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
).reset_index(drop=True)

print("\n==============================")
print("Model Comparison")
print("==============================")
print(results_df)

# ==========================================================
# Step 11 : Visualization
# ==========================================================

plt.figure(figsize=(8,5))

plt.bar(results_df["Model"], results_df["Accuracy"])

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Comparison")

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()

# ==========================================================
# Step 12 : Best Model
# ==========================================================

best_model_name = results_df.loc[0, "Model"]

print("\nBest Model :", best_model_name)

# ==========================================================
# Step 13 : Load Best Model
# ==========================================================

best_model = models[best_model_name]

# ==========================================================
# Step 14 : Predict New Student
# ==========================================================

new_student = pd.DataFrame(
    [[85, 78, 7, 8]],
    columns=[
        "Attendance",
        "Coding_Score",
        "Study_Hours",
        "Communication_Skills"
    ]
)

prediction = best_model.predict(new_student)

# ==========================================================
# Step 15 : Decode Prediction
# ==========================================================

result = encoder.inverse_transform(prediction)

print("Prediction :", result[0])