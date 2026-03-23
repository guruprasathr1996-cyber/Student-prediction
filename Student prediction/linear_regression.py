# ============================================================
# Algorithm : Linear Regression
# Project   : Early Warning System for At-Risk Students
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LinearRegression
from sklearn.metrics         import mean_squared_error, r2_score, accuracy_score

# 1. Load data
try:
    df = pd.read_csv("student_data.csv")
except FileNotFoundError:
    print("Error: 'student_data.csv' not found. Please run generate_dataset.py first.")
    exit()

# 2. Preprocessing
X = df.drop(columns=["At_Risk"])
y = df["At_Risk"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)

# 5. Evaluate
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

# Convert to binary for comparison
y_pred_bin = (y_pred >= 0.5).astype(int)
acc = accuracy_score(y_test, y_pred_bin)

print("=" * 40)
print("  LINEAR REGRESSION RESULTS")
print("=" * 40)
print(f"Mean Squared Error (MSE) : {mse:.4f}")
print(f"R² Score                 : {r2:.4f}")
print(f"Accuracy (as classifier) : {acc * 100:.2f}%")
print("=" * 40)
