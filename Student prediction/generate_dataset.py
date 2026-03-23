# ============================================================
# Script: generate_dataset.py
# Purpose: Generate a synthetic student dataset for the
#          Early Warning System ML project.
# Run this FIRST to create 'student_data.csv'
# ============================================================

import pandas as pd
import numpy as np

np.random.seed(42)
n = 500  # number of students

# --- Feature generation ---
attendance        = np.random.randint(40, 100, n)          # Attendance %
midterm_score     = np.random.randint(20, 100, n)          # Midterm exam score
final_score       = np.random.randint(20, 100, n)          # Final exam score
assignment_score  = np.random.randint(30, 100, n)          # Assignment average
study_hours       = np.random.randint(1, 10, n)            # Hours studied per week
participation     = np.random.randint(0, 10, n)            # Class participation score
failures          = np.random.randint(0, 4, n)             # Number of past failures
family_support    = np.random.randint(1, 5, n)             # Family support (1-5)
internet_access   = np.random.randint(0, 2, n)             # Internet at home (0/1)
extracurricular   = np.random.randint(0, 2, n)             # Extracurricular (0/1)

# --- Derive At_Risk label (rule-based for realism) ---
at_risk = (
    (attendance        < 65) |
    (midterm_score     < 40) |
    (final_score       < 40) |
    (assignment_score  < 45) |
    (failures          >= 2)  |
    (study_hours       < 3)
).astype(int)

# --- Build DataFrame ---
df = pd.DataFrame({
    "Attendance":        attendance,
    "Midterm_Score":     midterm_score,
    "Final_Score":       final_score,
    "Assignment_Score":  assignment_score,
    "Study_Hours":       study_hours,
    "Participation":     participation,
    "Failures":          failures,
    "Family_Support":    family_support,
    "Internet_Access":   internet_access,
    "Extracurricular":   extracurricular,
    "At_Risk":           at_risk,
})

df.to_csv("student_data.csv", index=False)
print(f"Dataset saved: student_data.csv  ({n} rows)")
print(f"At-Risk students: {at_risk.sum()} | Not At-Risk: {n - at_risk.sum()}")
