# ==========================================
# AI/ML Internship
# Day 1 - Python Logic & NumPy Basics
# Author: Ankit Raj
# ==========================================

# ==========================================
# Conditions in Python
# ==========================================

marks = 74

if marks >= 90:
    print("Excellent Performance")
elif marks >= 75:
    print("Good Performance")
else:
    print("Needs Improvement")

# ==========================================
# for Loop
# ==========================================

for i in range(5):
    print("AI/ML Internship")

# ==========================================
# range() Function
# ==========================================

print("\nNumbers from 1 to 6")

for number in range(1, 7):
    print(number)

print("\nEven Numbers")

for number in range(1, 6):
    if number % 2 == 0:
        print(number)

# ==========================================
# while Loop
# ==========================================

print("\nWhile Loop")

count = 1

while count <= 5:
    print(count)
    count += 1

# ==========================================
# Functions
# ==========================================

def welcome_student():
    print("\nWelcome to AI/ML Internship")

welcome_student()

# ==========================================
# Function with Parameters
# ==========================================

def student_marks(name, marks):
    print("Student Name:", name)
    print("Marks:", marks)

student_marks("Rahul", 85)

# ==========================================
# Function with Return
# ==========================================

def add_numbers(a, b):
    return a + b

result = add_numbers(5, 3)

print("\nAddition:", result)

# ==========================================
# Performance Checker Function
# ==========================================

def check_performance(marks):

    if marks >= 75:
        return "Good Performance"

    else:
        return "Needs Improvement"

print(check_performance(85))

# ==========================================
# Lists in Python
# ==========================================

marks = [80, 75, 90, 85]

print("\nOriginal List")

print(marks)

print("\nAccessing Elements")

print(marks[0])
print(marks[2])

marks.append(95)

print("\nAfter Append")

print(marks)

marks.remove(75)

print("\nAfter Remove")

print(marks)

# ==========================================
# Dictionaries in Python
# ==========================================

student = {
    "Name": "Rahul",
    "Marks": 85,
    "Attendance": 92
}

print("\nStudent Dictionary")

print(student)

print(student["Name"])
print(student["Attendance"])

# ==========================================
# NumPy Introduction
# ==========================================

import numpy as np

arr = np.array([10, 20, 30, 40])

print("\nNumPy Array")

print(arr)

# ==========================================
# NumPy Array Operations
# ==========================================

print("\nArray + 5")

print(arr + 5)

print("\nArray * 2")

print(arr * 2)

# ==========================================
# Mini Student Performance Program
# ==========================================

students_marks = [78, 90, 65, 88, 55]

print("\nStudent Performance")

for marks in students_marks:

    if marks >= 75:
        print(marks, "- Good Performance")

    else:
        print(marks, "- Needs Improvement")

# ==========================================
# End of Day 1
# ==========================================

print("\nDay 1 Completed Successfully!")