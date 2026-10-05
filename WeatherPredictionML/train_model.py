"""Trains the Decision Tree model (as described in the PBL report)."""
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

FEATURES = ["Temperature", "Humidity", "WindSpeed"]

df = pd.read_csv("dataset/weather.csv").dropna().drop_duplicates()
X, y = df[FEATURES], df["Weather"]

# 80:20 split, random_state=42  (Section 3.4 of the report)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Iteration 1 - baseline
base = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
base_acc = accuracy_score(y_test, base.predict(X_test))

# Iteration 2 - refinement (limit depth to reduce over-fitting)
model = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train, y_train)
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

classes = list(model.classes_)
metrics = {
    "baseline_accuracy": round(base_acc * 100, 2),
    "accuracy": round(acc * 100, 2),
    "rows": int(len(df)),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "classes": classes,
    "class_counts": {c: int((y == c).sum()) for c in classes},
    "confusion_matrix": confusion_matrix(y_test, pred, labels=classes).tolist(),
    "feature_importance": {f: round(float(i), 4) for f, i in zip(FEATURES, model.feature_importances_)},
    "report": classification_report(y_test, pred, output_dict=True),
}

joblib.dump(model, "model/weather_model.pkl")
json.dump(metrics, open("model/metrics.json", "w"), indent=2)
open("model/tree_rules.txt", "w").write(export_text(model, feature_names=FEATURES))

print(f"Baseline accuracy (Iteration 1): {metrics['baseline_accuracy']}%")
print(f"Refined accuracy  (Iteration 2): {metrics['accuracy']}%")
print(classification_report(y_test, pred))
print("Model saved to model/weather_model.pkl")
