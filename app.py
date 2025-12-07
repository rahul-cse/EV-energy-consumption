# -*- coding: utf-8 -*-
"""
@author: rahul
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import tensorflow as tf
from joblib import load
from fastapi.middleware.cors import CORSMiddleware


trained_model = tf.keras.models.load_model("ev_energy_model.keras")


app = FastAPI("title="EV Energy Consumption API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# input schema
class EVFeatures(BaseModel):
    Speed_kmh: float
    Acceleration_ms2: float
    Battery_State_: float
    Battery_Voltage_V: float
    Battery_Temperature_C: float
    Driving_Mode: int = Field(..., ge=1, le=3) 
    Road_Type: int = Field(..., ge=1, le=3)
    Traffic_Condition: int = Field(..., ge=1, le=3)
    Weather_Condition: int = Field(..., ge=1, le=4)
    Slope_: float
    Temperature_C: float
    Humidity_: float
    Wind_Speed_ms: float
    Tire_Pressure_psi: float
    Vehicle_Weight_kg: float
    Distance_Travelled_km: float
    

@app.post("/predict")
def predict_energy(data: EVFeatures):
    
    dm = one_hot_encode(data.Driving_Mode, 3)

    rt = one_hot_encode(data.Road_Type, 3)

    tc = one_hot_encode(data.Traffic_Condition, 3)

    wc = one_hot_encode(data.Weather_Condition, 4)

    numeric_features = [
                   data.Speed_kmh,
                   data.Acceleration_ms2,
                   data.Battery_State_,
                   data.Battery_Voltage_V,
                   data.Battery_Temperature_C,
                   data.Slope_,
                   data.Temperature_C,
                   data.Humidity_,
                   data.Wind_Speed_ms,
                   data.Tire_Pressure_psi,
                   data.Vehicle_Weight_kg,
                   data.Distance_Travelled_km,
                  ]
    
    x = np.array([numeric_features + dm + rt + tc + wc], dtype=float)
    scaler_X = load("scaler_X.save")
    x_scaled = scaler_X.transform(x)
    
    # Prediction
    pred = trained_model.predict(x_scaled)
    return {"Energy_Consumption_kWh": float(pred[0][0])}    

def one_hot_encode(value, num_categories):
    arr = [0] * num_categories
    arr[value-1] = 1
    return arr