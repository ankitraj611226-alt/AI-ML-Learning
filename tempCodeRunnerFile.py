import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
student_data={
  'Student_ID':[],
  'Attendance':[],
  'coding_score':[],
  'study_Hours':[],
  'Communication_Skills':[],
  'Placement_Status':[]
}
for i in range(100):
  student_data['Student_ID'].append(i+1)
  student_data['Attendance'].append(random.randint(40,100))
  student_data['coding_score'].append(
    random.randint(20,100)
  )
  student_data['study_Hours'].append(random.randint(1,10))
  student_data['Communication_Skills'].append(random.randint(1,10))
  student_data['Placement_Status'].append(random.choice(['Placed','Not Placed']))
df=pd.DataFrame(student_data)
print(df.head())
df.loc[5,'coding_score']=np.nan
df.loc[12,'Attendance']=np.nan
df.loc[20,'Placement_Status']=np.nan
df=pd.concat([df,df.iloc[[0]]],ignore_index=True)
print(df.shape)
df.loc[10,'coding_score']=500
print(df.loc[10])
df["coding_score"] = df["coding_score"].fillna(
    df["coding_score"].mean()
)
print(df.isnull().sum())
df["coding_score"]=df["coding_score"].fillna(
  df["coding_score"].median()
)
print(df.isnull().sum())
df["Placement_Status"].mode()#df["Placement_Status"].mode()[0] us list ki pheli value nikalta hai.
df["Placement_Status"]=df["Placement_Status"].fillna(df["Placement_Status"].mode()[0])
print(df.duplicated())
print(df.duplicated().sum())
df=df.drop_duplicates()
print(df.shape)
q1=df["coding_score"].quantile(0.25)
q3=df["coding_score"].quantile(0.75)
IQR=q3-q1
lower=q1-1.5*IQR
Upper=q3+1.5*IQR
outliers=df[(df["coding_score"]<lower)|(df["coding_score"]>Upper)]
print(outliers)
print("Q1 =", q1)
print("Q3 =", q3)
print("IQR =", IQR)
print("Lower Limit =", lower)
print("Upper Limit =", Upper)

print("\nOutliers:")
print(outliers)
df=df[(df["coding_score"]>=lower)&(df["coding_score"]<=Upper)]
print(df)
print(df.shape)
scaler=MinMaxScaler()
df["coding_score"]=scaler.fit_transform(df[["coding_score"]])
print(df["coding_score"])
df["coding_score"]=scaler.fit_transform(df[["coding_score"]])
print(df["coding_score"])
encoder = LabelEncoder()
df["Placement_Status"] = encoder.fit_transform(df["Placement_Status"])
print(df["Placement_Status"])
df["Branch"] = [
    "CSE",
    "ECE",
    "ME",
    "AI",
] *25
print(df)
















  
  
  
  
   