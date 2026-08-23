from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# Application
# --------------------------------------------------

app = FastAPI(
    title="Device Repair vs Replace API",
    description="Predict whether a device should be repaired or replaced using Decision Tree.",
    version="1.0.0"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Load ML Model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "decision_tree.pkl"

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Input Schema
# --------------------------------------------------

class DeviceData(BaseModel):

    device_age_years: float = Field(
        ...,
        ge=0.1,
        le=10,
        description="Age of the device in years"
    )

    battery_health: float = Field(
        ...,
        ge=0,
        le=100,
        description="Battery health percentage"
    )

    screen_damage: int = Field(
        ...,
        ge=0,
        le=1,
        description="0 = No, 1 = Yes"
    )

    body_condition: int = Field(
        ...,
        ge=0,
        le=2,
        description="0 = Good, 1 = Average, 2 = Poor"
    )

    repair_cost: float = Field(
        ...,
        ge=0,
        description="Estimated repair cost in INR"
    )

    device_value: float = Field(
        ...,
        ge=0,
        description="Current estimated device value in INR"
    )

    previous_repairs: int = Field(
        ...,
        ge=0,
        le=20,
        description="Number of previous repairs"
    )

    warranty_remaining: int = Field(
        ...,
        ge=0,
        le=1,
        description="0 = No, 1 = Yes"
    )


# --------------------------------------------------
# Home
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Device Repair vs Replace API is running"
    }


# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.post("/predict")
def predict(device: DeviceData):

    input_data = pd.DataFrame([{
        "device_age_years": device.device_age_years,
        "battery_health": device.battery_health,
        "screen_damage": device.screen_damage,
        "body_condition": device.body_condition,
        "repair_cost": device.repair_cost,
        "device_value": device.device_value,
        "previous_repairs": device.previous_repairs,
        "warranty_remaining": device.warranty_remaining
    }])

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    if prediction == 0:
        recommendation = "REPAIR"
    else:
        recommendation = "REPLACE"

    confidence = max(probabilities) * 100

    return {
        "recommendation": recommendation,
        "confidence": round(confidence, 2)
    }