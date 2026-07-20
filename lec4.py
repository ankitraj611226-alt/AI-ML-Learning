# # import pandas as pd
# # df=pd.read_csv("student.csv")
# # print(df.info())
# # #import pandas as pd

# # student_data = {
# #     "Name": ["Rahul", "Aman", "Priya"],
# #     "Marks": [85, 76, 91],
# #     "Attendance": [92, 81, 95]
# # }

# # df = pd.DataFrame(student_data)

# # print(df.describe())
# # print(df.isnull())
# # print(df.isnull().sum())
# # # import pandas as pd

# # student_data = {
# #     "Name": ["Rahul", "Aman", "Priya"],
# #     "Marks": [85, None, 91],
# #     "Attendance": [92, 81, 95]
# # }

# # df = pd.DataFrame(student_data)

# # print(df.isnull().sum())
# # import pandas as pd

# # student_data = {
# #     "Gender": ["Male", "Male", "Female", "Female", "Male"]
# # }

# # df = pd.DataFrame(student_data)
# # print(df["Gender"].value_counts())
# # print(df["Gender"].unique)
# # df["Gender"].nunique()
# # import pandas as pd

# # student_data = {
# #     "Name": ["Rahul", "Aman", "Priya", "Neha", "Karan", "Riya"],
# #     "Branch": ["CSE", "AI/ML", "ECE", "CSE", "Mechanical", "AI/ML"]
# # }

# # df = pd.DataFrame(student_data)
# # print(df["Branch"].unique())
# # print(df["Branch"].nunique())
# # print(df["Branch"].value_counts())
# #import pandas as pd

# # student_data = {
# #     "Study_Hours": [2, 4, 6, 8],
# #     "Marks": [50, 70, 90, 98]
# # }

# # df = pd.DataFrame(student_data)

# #print(df.corr(numeric_only=True))
# # student_data={
# #   "Study_Hours":[2,4,6,8],
# #   "Marks":[50,60,70,90]
# # }
# # df=pd.DataFrame(student_data)
# # print(df.corr(numeric_only=True))
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns

# # student_data = {
#     "Study_Hours": [2, 4, 6, 8],
#     "Marks": [50, 70, 90, 98],
#     "Attendance": [60, 75, 90, 95]
# }

# df = pd.DataFrame(student_data)

# sns.heatmap(df.corr(numeric_only=True), annot=True)

# plt.show()
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# student_data = {
#     "Study_Hours": [2,4,6,8,10,12],
#     "Marks": [50,70,90,98,89,78],
#     "Attendance": [60,75,90,91,92,93]
# }

# df = pd.DataFrame(student_data)
# plt.hist(df["Attendance"])
# plt.xlabel("Attendance")
# plt.ylabel("Count")
# plt.title("Attendance Distribution")
# plt.show()
# sns.boxplot(x=df["soft_skill_score"])
# plt.show()
import pandas as pd

student_data = {
    "Soft_Skill_Score": [50, 55, 60, 65, 70, 75, 80, 85, 90, 150]
}

df = pd.DataFrame(student_data)
import matplotlib.pyplot as plt
import seaborn as sns

sns.boxplot(x=df["Soft_Skill_Score"])

plt.title("Soft Skill Score")
plt.show()



