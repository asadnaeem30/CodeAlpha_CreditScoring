"""
Credit Scoring Model
CodeAlpha Internship | Task 1
Author: Your Name
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report,
                             roc_curve, auc)

print("=" * 60)
print("       Credit Scoring Model — CodeAlpha Task 1")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. CREATE DATASET
# ─────────────────────────────────────────────
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'Age': np.random.randint(18, 75, n),
    'Sex': np.random.choice(['male', 'female'], n),
    'Job': np.random.randint(0, 4, n),
    'Housing': np.random.choice(['own', 'free', 'rent'], n),
    'Saving accounts': np.random.choice(['little', 'moderate',
                                         'rich', 'quite rich'], n),
    'Checking account': np.random.choice(['little', 'moderate',
                                          'rich'], n),
    'Credit amount': np.random.randint(500, 15000, n),
    'Duration': np.random.randint(6, 72, n),
    'Purpose': np.random.choice(['car', 'furniture', 'radio/TV',
                                 'education', 'business'], n),
    'Risk': np.random.choice(['good', 'bad'], n, p=[0.7, 0.3])
})

print(f"\n[INFO] Dataset shape : {df.shape}")

# ─────────────────────────────────────────────
# 2. PREPROCESS
# ─────────────────────────────────────────────
df['Saving accounts']  = df['Saving accounts'].fillna('unknown')
df['Checking account'] = df['Checking account'].fillna('unknown')

le = LabelEncoder()
for col in ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']:
    df[col] = le.fit_transform(df[col])

df['Risk'] = (df['Risk'] == 'good').astype(int)
print("[INFO] Data preprocessed!")

# ─────────────────────────────────────────────
# 3. SPLIT DATA
# ─────────────────────────────────────────────
X = df.drop('Risk', axis=1)
y = df['Risk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"[INFO] Training samples : {len(X_train)}")
print(f"[INFO] Testing  samples : {len(X_test)}")

# ─────────────────────────────────────────────
# 4. TRAIN MODELS
# ─────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'Random Forest':       RandomForestClassifier(random_state=42),
    'XGBoost':             XGBClassifier(random_state=42)
}

print("\n[INFO] Training models...")
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"  ✅ {name} trained!")

# ─────────────────────────────────────────────
# 5. EVALUATE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"{'Model':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
print("=" * 60)

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"{name:<25} {accuracy_score(y_test,y_pred):>6.2f} "
          f"{precision_score(y_test,y_pred):>6.2f} "
          f"{recall_score(y_test,y_pred):>6.2f} "
          f"{f1_score(y_test,y_pred):>6.2f} "
          f"{roc_auc_score(y_test,y_prob):>6.2f}")

print("=" * 60)

# ─────────────────────────────────────────────
# 6. CONFUSION MATRIX
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Confusion Matrix — All Models", fontsize=14, fontweight="bold")

for ax, (name, model) in zip(axes.flatten(), models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Bad", "Good"],
                yticklabels=["Bad", "Good"], ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("[INFO] Saved: confusion_matrix.png")

# ─────────────────────────────────────────────
# 7. ROC CURVE
# ─────────────────────────────────────────────
plt.figure(figsize=(10, 6))
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.2f})")

plt.plot([0,1],[0,1],'k--')
plt.title("ROC Curve — All Models", fontsize=14, fontweight="bold")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig("roc_curve.png", dpi=150)
plt.show()
print("[INFO] Saved: roc_curve.png")

# ─────────────────────────────────────────────
# 8. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
feature_names = df.drop('Risk', axis=1).columns
importances   = models['Random Forest'].feature_importances_
indices       = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices], color="#2196F3")
plt.xticks(range(len(importances)),
           [feature_names[i] for i in indices], rotation=45)
plt.title("Feature Importance — Random Forest", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
print("[INFO] Saved: feature_importance.png")
print("\n🎉 Credit Scoring Model Complete!")
