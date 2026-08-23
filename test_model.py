import joblib
import pandas as pd

# Load trained model
model = joblib.load("model/decision_tree.pkl")

# New device details
device = pd.DataFrame([{
    "device_age_years": 4.2,
    "battery_health": 55,
    "screen_damage": 1,
    "body_condition": 2,
    "repair_cost": 18000,
    "device_value": 15000,
    "previous_repairs": 3,
    "warranty_remaining": 0
}])

# Make prediction
prediction = model.predict(device)[0]

# Get prediction probability
probabilities = model.predict_proba(device)[0]

# Convert result to readable text
if prediction == 0:
    result = "REPAIR"
else:
    result = "REPLACE"

confidence = max(probabilities) * 100

print("Device Repair vs Replace Prediction")
print("------------------------------------")
print(f"Recommendation : {result}")
print(f"Confidence     : {confidence:.2f}%")