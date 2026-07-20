"""
=========================================================
Lecture 08

Topic:
Feature Engineering
=========================================================
"""

import numpy as np
import pandas as pd

# ==========================================
# Create Dataset
# ==========================================

np.random.seed(42)

student_data = {

    "Coding_Score": np.random.randint(40,101,10),

    "DSA_Score": np.random.randint(35,101,10),

    "Python_Score": np.random.randint(40,101,10),

    "SQL_Score": np.random.randint(35,101,10),

    "Projects": np.random.randint(0,6,10),

    "Certification_Count": np.random.randint(0,5,10),

    "Hackathon_Participation": np.random.randint(0,5,10),

    "Resume_Score": np.random.randint(50,101,10),

    "Communication_Skills": np.random.randint(5,11,10),

    "Mock_Interview": np.random.randint(5,11,10)

}

df = pd.DataFrame(student_data)

print("Original Dataset\n")

print(df)

# ==========================================
# Feature Engineering
# ==========================================

df["Technical_Score"] = (

    df["Coding_Score"]

    + df["DSA_Score"]

    + df["Python_Score"]

    + df["SQL_Score"]

) / 4

df["Profile_Score"] = (

    df["Projects"] * 10

    + df["Certification_Count"] * 5

    + df["Hackathon_Participation"] * 8

    + df["Resume_Score"]

) / 4

df["Interview_Score"] = (

    df["Communication_Skills"] * 10

    + df["Mock_Interview"] * 10

) / 2

print("\n\nNew Features\n")

print(

    df[

        [

            "Technical_Score",

            "Profile_Score",

            "Interview_Score"

        ]

    ]

)