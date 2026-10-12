"""
Credit Card Fraud Detection using Logistic Regression
-----------------------------------------------------
Dataset : creditcard_dataset.csv  (284,807 transactions, 31 columns)
          Time, V1..V28 (anonymised PCA features), Amount, Class (0 = legit, 1 = fraud)

Workflow (same as the project guide):
  1. Collect data          5. Evaluate (precision, recall, F1, confusion matrix, PR-AUC)
  2. Prepare data          6. Choose a threshold (on a validation set)
  3. Handle class imbalance 7. Monitor / improve (feature importance, sample scoring)
  4. Train Logistic Regression

Run in Jupyter / Google Colab / terminal.
In Colab: upload creditcard_dataset.csv first (left sidebar -> Files).

Install:  pip install pandas numpy scikit-learn matplotlib seaborn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

RANDOM_STATE = 42
DATA_PATH = "creditcard_dataset.csv"   # change if your file is elsewhere


# ----------------------------------------------------------------------
# 1. COLLECT DATA
# ----------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

print("--- Dataset overview ---")
print("Shape:", df.shape)
print(df[["Time", "Amount", "Class"]].head())


# ----------------------------------------------------------------------
# 2. PREPARE DATA
# ----------------------------------------------------------------------
print("\nMissing values :", df.isnull().sum().sum())
print("Duplicate rows :", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

# Class balance
counts = df["Class"].value_counts()
print("\n--- Class balance ---")
print(f"Legitimate (0): {counts[0]}")
print(f"Fraud      (1): {counts[1]}  ({counts[1] / len(df) * 100:.3f}%)")

plt.figure(figsize=(5, 4))
sns.countplot(x="Class", data=df, hue="Class", palette="Set2", legend=False)
plt.yscale("log")  # log scale so the tiny fraud bar is still visible
plt.title("Class Distribution (0 = Legitimate, 1 = Fraud)")
plt.tight_layout()
plt.savefig("class_balance.png", dpi=120)
plt.show()

# Features (X) and label (y)
X = df.drop(columns=["Class"])
y = df["Class"]

# Chronological split (the data is time-ordered, so we train on the past and
# test on the future, exactly like a real bank would):
#   first 70%  -> train      next 15% -> validation (to pick threshold)
#   last 15%   -> test (never touched until the final evaluation)
df_sorted_idx = X["Time"].sort_values(kind="stable").index
n = len(df)
i_train, i_val = int(n * 0.70), int(n * 0.85)

train_idx = df_sorted_idx[:i_train]
val_idx = df_sorted_idx[i_train:i_val]
test_idx = df_sorted_idx[i_val:]

X_train, y_train = X.loc[train_idx], y.loc[train_idx]
X_val, y_val = X.loc[val_idx], y.loc[val_idx]
X_test, y_test = X.loc[test_idx], y.loc[test_idx]

print("\n--- Chronological split ---")
for name, yy in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
    print(f"{name:<11}: {len(yy):>7} rows | fraud = {yy.sum():>3} ({yy.mean() * 100:.3f}%)")


# ----------------------------------------------------------------------
# 3 & 4. HANDLE IMBALANCE + TRAIN LOGISTIC REGRESSION
# ----------------------------------------------------------------------
# * V1..V28 are already PCA-scaled, but Time and Amount are not, so we scale
#   every feature with StandardScaler. Inside a Pipeline the scaler learns
#   only from TRAINING data (no data leakage).
# * class_weight="balanced" makes fraud mistakes count much more than
#   legitimate mistakes, so the model does not simply predict "legit" always.
#   The validation/test sets are NOT rebalanced - they stay realistic.
model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

model.fit(X_train, y_train)
print("\nLogistic Regression model trained successfully.")


# ----------------------------------------------------------------------
# 5. EVALUATE ON THE TEST SET (default threshold = 0.5)
# ----------------------------------------------------------------------
y_prob = model.predict_proba(X_test)[:, 1]      # fraud probability = risk score
y_pred = (y_prob >= 0.5).astype(int)

print("\n--- Classification report (test set, threshold = 0.5) ---")
print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"], digits=4))
print("ROC-AUC             :", round(roc_auc_score(y_test, y_prob), 4))
print("PR-AUC (Avg. Prec.) :", round(average_precision_score(y_test, y_prob), 4))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\nTN = {tn} | FP = {fp} (false alarms) | FN = {fn} (missed fraud) | TP = {tp}")

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Pred Legit", "Pred Fraud"],
    yticklabels=["Actual Legit", "Actual Fraud"],
)
plt.title("Confusion Matrix (threshold = 0.5)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
plt.show()

# Precision-Recall curve (more informative than ROC when fraud is rare)
prec, rec, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(6, 4))
plt.plot(rec, prec, label=f"Logistic Regression (AP = {average_precision_score(y_test, y_prob):.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (test set)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("precision_recall_curve.png", dpi=120)
plt.show()


# ----------------------------------------------------------------------
# 6. CHOOSE A THRESHOLD (using the VALIDATION set, not the test set)
# ----------------------------------------------------------------------
# A lower threshold catches more fraud but creates more false alerts.
val_prob = model.predict_proba(X_val)[:, 1]
p_v, r_v, t_v = precision_recall_curve(y_val, val_prob)
f1_v = 2 * p_v[:-1] * r_v[:-1] / (p_v[:-1] + r_v[:-1] + 1e-12)
best_threshold = float(t_v[np.argmax(f1_v)])
print(f"\nBest-F1 threshold chosen on validation set: {best_threshold:.6f}")

print("\n--- Test-set results at different thresholds ---")
print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Alerts':>8}")
for t in [0.5, 0.7, 0.9, 0.95, best_threshold]:
    pred_t = (y_prob >= t).astype(int)
    tag = "  <- chosen" if t == best_threshold else ""
    print(
        f"{t:>10.6f} "
        f"{precision_score(y_test, pred_t, zero_division=0):>10.3f} "
        f"{recall_score(y_test, pred_t):>8.3f} "
        f"{f1_score(y_test, pred_t, zero_division=0):>8.3f} "
        f"{pred_t.sum():>8d}{tag}"
    )

# Final predictions with the chosen threshold
final_pred = (y_prob >= best_threshold).astype(int)
print("\n--- Final classification report (chosen threshold) ---")
print(classification_report(y_test, final_pred, target_names=["Legit", "Fraud"], digits=4))

cm2 = confusion_matrix(y_test, final_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm2, annot=True, fmt="d", cmap="Greens",
    xticklabels=["Pred Legit", "Pred Fraud"],
    yticklabels=["Actual Legit", "Actual Fraud"],
)
plt.title(f"Confusion Matrix (threshold = {best_threshold:.4f})")
plt.tight_layout()
plt.savefig("confusion_matrix_tuned.png", dpi=120)
plt.show()


# ----------------------------------------------------------------------
# 7. UNDERSTAND THE MODEL + SCORE A TRANSACTION
# ----------------------------------------------------------------------
# Logistic regression is interpretable: the coefficient shows how strongly
# each (scaled) feature pushes the fraud risk up (+) or down (-).
coefs = pd.Series(model.named_steps["clf"].coef_[0], index=X.columns)
top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(10)

plt.figure(figsize=(6, 4))
top.sort_values().plot(kind="barh", color=np.where(top.sort_values() > 0, "tomato", "steelblue"))
plt.title("Top 10 Features by Influence (coefficients)")
plt.xlabel("Coefficient (red = raises fraud risk)")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120)
plt.show()

# Score one example fraud transaction from the test set
fraud_rows = X_test[y_test == 1]
sample = fraud_rows.iloc[[0]]
risk = model.predict_proba(sample)[0, 1]
decision = "POTENTIAL FRAUD -> send for review" if risk >= best_threshold else "Looks normal"
print(f"\nSample transaction risk score: {risk:.4f} -> {decision}")
print("Actual label: Fraud")

print("\nNote: the model gives a risk score to support a decision; it does not "
      "prove fraud. A human analyst / extra verification should review alerts.")