import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# 1. Load dataset
df = pd.read_csv("dataset/device_repair_dataset.csv")

# 2. Separate features and target
X = df.drop("repair_or_replace", axis=1)
y = df["repair_or_replace"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Create Decision Tree model
model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

# 5. Train model
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("Model trained successfully!")
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["REPAIR", "REPLACE"]
))

# 8. Save model
joblib.dump(model, "model/decision_tree.pkl")

print("\nModel saved to:")
print("model/decision_tree.pkl")