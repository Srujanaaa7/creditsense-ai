import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# ─── Load Dataset ────────────────────────────────────────────────────────────
df = pd.read_csv("credit_risk_dataset.csv")

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {df.columns.tolist()}")

# ─── Clean Data ──────────────────────────────────────────────────────────────
df = df.dropna()

# One-hot encode categorical columns
df = pd.get_dummies(df, drop_first=True)

# ─── Target Column ───────────────────────────────────────────────────────────
target_col = "loan_status"

if target_col not in df.columns:
    raise ValueError(f"Column '{target_col}' not found. Available: {df.columns.tolist()}")

# Convert text labels if needed
if df[target_col].dtype == 'object':
    df[target_col] = df[target_col].map({'Fully Paid': 0, 'Charged Off': 1})

X = df.drop(target_col, axis=1)
y = df[target_col]

# ─── Train/Test Split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── Scale Features ──────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─── Train Model ─────────────────────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# ─── Evaluate ────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ─── Save Artifacts ──────────────────────────────────────────────────────────
pickle.dump(model,   open("model.pkl",   "wb"))
pickle.dump(scaler,  open("scaler.pkl",  "wb"))
pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))

print("\n✅ model.pkl, scaler.pkl, columns.pkl saved successfully!")
