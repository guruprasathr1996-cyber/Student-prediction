# ============================================================
# Project : Early Warning System for Identifying At-Risk Students
#           Using Machine Learning
# Dataset : student_data.csv  (target column = "At_Risk")
# Models  : Linear Regression, Logistic Regression,
#           Decision Tree, Random Forest
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model    import LinearRegression, LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier

# ─────────────────────────────────────────────────────────────
# SECTION 1 — DATA PREPROCESSING
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  SECTION 1: DATA PREPROCESSING")
print("=" * 60)

# Load dataset
df = pd.read_csv("student_data.csv")
print(f"\nDataset loaded successfully — Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values
missing = df.isnull().sum()
print(f"\nMissing values per column:\n{missing}")
df.dropna(inplace=True)          # drop rows with any missing value
print(f"\nShape after handling missing values: {df.shape}")

# Separate features (X) and target (y)
X = df.drop(columns=["At_Risk"])
y = df["At_Risk"]

print(f"\nFeatures (X) shape : {X.shape}")
print(f"Target   (y) shape : {y.shape}")
print(f"Class distribution :\n{y.value_counts().rename({0: 'Not At Risk', 1: 'At Risk'})}")

# Train-test split (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")


# ─────────────────────────────────────────────────────────────
# SECTION 2a — LINEAR REGRESSION
# (Treated as a regression task for demonstration)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SECTION 2a: LINEAR REGRESSION")
print("=" * 60)

# Train
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict
y_pred_lr = lr_model.predict(X_test)

# Evaluate
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr  = r2_score(y_test, y_pred_lr)

print(f"\nMean Squared Error (MSE) : {mse_lr:.4f}")
print(f"R² Score                 : {r2_lr:.4f}")

# Convert continuous output → binary for comparison purposes
y_pred_lr_bin = (y_pred_lr >= 0.5).astype(int)
acc_lr = accuracy_score(y_test, y_pred_lr_bin)
print(f"Accuracy (threshold=0.5) : {acc_lr * 100:.2f}%")


# ─────────────────────────────────────────────────────────────
# SECTION 2b — LOGISTIC REGRESSION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SECTION 2b: LOGISTIC REGRESSION")
print("=" * 60)

# Train
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)

# Predict
y_pred_log = log_model.predict(X_test)

# Evaluate
acc_log  = accuracy_score (y_test, y_pred_log)
prec_log = precision_score(y_test, y_pred_log)
rec_log  = recall_score   (y_test, y_pred_log)
f1_log   = f1_score       (y_test, y_pred_log)
cm_log   = confusion_matrix(y_test, y_pred_log)

print(f"\nAccuracy  : {acc_log  * 100:.2f}%")
print(f"Precision : {prec_log * 100:.2f}%")
print(f"Recall    : {rec_log  * 100:.2f}%")
print(f"F1-Score  : {f1_log   * 100:.2f}%")
print(f"\nConfusion Matrix:\n{cm_log}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log,
                             target_names=["Not At Risk", "At Risk"]))


# ─────────────────────────────────────────────────────────────
# SECTION 2c — DECISION TREE CLASSIFIER
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SECTION 2c: DECISION TREE CLASSIFIER")
print("=" * 60)

# Train
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)

# Predict
y_pred_dt = dt_model.predict(X_test)

# Evaluate
acc_dt  = accuracy_score (y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt)
rec_dt  = recall_score   (y_test, y_pred_dt)
f1_dt   = f1_score       (y_test, y_pred_dt)
cm_dt   = confusion_matrix(y_test, y_pred_dt)

print(f"\nAccuracy  : {acc_dt  * 100:.2f}%")
print(f"Precision : {prec_dt * 100:.2f}%")
print(f"Recall    : {rec_dt  * 100:.2f}%")
print(f"F1-Score  : {f1_dt   * 100:.2f}%")
print(f"\nConfusion Matrix:\n{cm_dt}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt,
                             target_names=["Not At Risk", "At Risk"]))


# ─────────────────────────────────────────────────────────────
# SECTION 2d — RANDOM FOREST CLASSIFIER
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SECTION 2d: RANDOM FOREST CLASSIFIER")
print("=" * 60)

# Train
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict
y_pred_rf = rf_model.predict(X_test)

# Evaluate
acc_rf  = accuracy_score (y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
rec_rf  = recall_score   (y_test, y_pred_rf)
f1_rf   = f1_score       (y_test, y_pred_rf)
cm_rf   = confusion_matrix(y_test, y_pred_rf)

print(f"\nAccuracy  : {acc_rf  * 100:.2f}%")
print(f"Precision : {prec_rf * 100:.2f}%")
print(f"Recall    : {rec_rf  * 100:.2f}%")
print(f"F1-Score  : {f1_rf   * 100:.2f}%")
print(f"\nConfusion Matrix:\n{cm_rf}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf,
                             target_names=["Not At Risk", "At Risk"]))

# Feature Importance
feat_importances = pd.Series(
    rf_model.feature_importances_, index=X.columns
).sort_values(ascending=False)
print("\nTop Feature Importances (Random Forest):")
print(feat_importances.to_string())


# ─────────────────────────────────────────────────────────────
# SECTION 3 — MODEL COMPARISON TABLE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SECTION 3: MODEL COMPARISON TABLE")
print("=" * 60)

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression (binary)",
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
    ],
    "Accuracy (%)": [
        round(acc_lr  * 100, 2),
        round(acc_log * 100, 2),
        round(acc_dt  * 100, 2),
        round(acc_rf  * 100, 2),
    ],
    "Precision (%)": [
        round(precision_score(y_test, y_pred_lr_bin) * 100, 2),
        round(prec_log * 100, 2),
        round(prec_dt  * 100, 2),
        round(prec_rf  * 100, 2),
    ],
    "Recall (%)": [
        round(recall_score(y_test, y_pred_lr_bin) * 100, 2),
        round(rec_log * 100, 2),
        round(rec_dt  * 100, 2),
        round(rec_rf  * 100, 2),
    ],
    "F1-Score (%)": [
        round(f1_score(y_test, y_pred_lr_bin) * 100, 2),
        round(f1_log * 100, 2),
        round(f1_dt  * 100, 2),
        round(f1_rf  * 100, 2),
    ],
})

# Additional regression metrics for Linear Regression
print(f"\n  [Linear Regression — Regression Metrics]")
print(f"  MSE  = {mse_lr:.4f}")
print(f"  R²   = {r2_lr:.4f}")

print()
print(comparison.to_string(index=False))

# Identify best model (by Accuracy among classifiers)
best_idx   = comparison["Accuracy (%)"].idxmax()
best_model = comparison.loc[best_idx, "Model"]
best_acc   = comparison.loc[best_idx, "Accuracy (%)"]

print(f"\n  🏆 Best Performing Model : {best_model}")
print(f"     Accuracy             : {best_acc}%")


# ─────────────────────────────────────────────────────────────
# SECTION 4 — CONCLUSION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SECTION 4: CONCLUSION")
print("=" * 60)
print(f"""
Based on the evaluation metrics:

• Linear Regression is NOT ideal for classification tasks because
  it predicts continuous values (not probabilities or classes).
  It is included here only to demonstrate its limitations when
  applied to a binary classification problem.

• Logistic Regression is a strong baseline classifier. It is
  simple, interpretable, and works well when features are
  linearly separable.

• Decision Tree Classifier is easy to interpret visually and
  handles non-linear relationships better than linear models,
  but it may overfit without depth constraints.

• Random Forest Classifier is typically the best performer
  because it builds multiple decision trees and averages their
  predictions, reducing overfitting and improving robustness.

★ Best Model: {best_model}
  Reason: Random Forest combines the predictions of several
  decision trees (ensemble learning), making it more accurate
  and less prone to overfitting than a single tree. It also
  provides feature importance rankings, which helps educators
  understand WHICH factors most strongly predict student risk.
""")
print("=" * 60)
print("  ✅ Project Complete — Early Warning System for At-Risk Students")
print("=" * 60)
