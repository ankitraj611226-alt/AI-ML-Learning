#Practie question
student_data={
  "name":["Rahul","Aman","Priya"],
  "marks":[56,67,89],
  "Attendence":[45,67,89]
}
print(student_data)
import pandas as pd
df=pd.DataFrame(student_data)
print(df)
print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
print(df['marks'])
print(pd.read_csv())
