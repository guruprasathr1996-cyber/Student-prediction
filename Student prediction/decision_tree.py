# ============================================================
# Algorithm : Decision Tree Classifier
# Project   : Early Warning System for At-Risk Students
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree            import DecisionTreeClassifier
from sklearn.metrics         import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Load data
try:
    df = pd.read_csv("student_data.csv")
except FileNotFoundError:
    print("Error: 'student_data.csv' not found. Please run generate_dataset.py first.")
    exit()

# 2. Preprocessing
X = df.drop(columns=["At_Risk"])
y = df["At_Risk"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Train Model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)

# 5. Evaluate
print("=" * 40)
print("  DECISION TREE RESULTS")
print("=" * 40)
print(f"Accuracy  : {accuracy_score (y_test, y_pred) * 100:.2f}%")
print(f"Precision : {precision_score(y_test, y_pred) * 100:.2f}%")
print(f"Recall    : {recall_score   (y_test, y_pred) * 100:.2f}%")
print(f"F1-Score  : {f1_score       (y_test, y_pred) * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("=" * 40)
