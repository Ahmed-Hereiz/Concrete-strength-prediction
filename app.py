# app.py - FastAPI Backend

import sys
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict
from preprocessing import create_engineered_features  # Assuming this module is available

# Define features, short names, units, ranges (same as in gui.py)
features = [
    'Cement (component 1)(kg in a m^3 mixture)',
    'Blast Furnace Slag (component 2)(kg in a m^3 mixture)',
    'Fly Ash (component 3)(kg in a m^3 mixture)',
    'Water  (component 4)(kg in a m^3 mixture)',
    'Superplasticizer (component 5)(kg in a m^3 mixture)',
    'Coarse Aggregate  (component 6)(kg in a m^3 mixture)',
    'Fine Aggregate (component 7)(kg in a m^3 mixture)',
    'Age (day)'
]

short_names = [
    'Cement',
    'Blast Furnace Slag', 
    'Fly Ash',
    'Water',
    'Superplasticizer',
    'Coarse Aggregate',
    'Fine Aggregate',
    'Age'
]

units = [
    'kg/m³', 'kg/m³', 'kg/m³', 'kg/m³', 'kg/m³', 'kg/m³', 'kg/m³', 'days'
]

ranges = [
    (90.0, 600.0, 300.0),  # Cement
    (0.0, 400.0, 100.0),   # Slag
    (0.0, 220.0, 50.0),    # Fly Ash
    (110.0, 270.0, 180.0), # Water
    (0.0, 35.0, 5.0),      # Superplasticizer
    (750.0, 1200.0, 1000.0), # Coarse Aggregate
    (550.0, 1050.0, 800.0), # Fine Aggregate
    (1.0, 400.0, 28.0)     # Age
]

app = FastAPI()

# Mount static files (for CSS and JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load models at startup
models = {}
model_files = {
    'Random Forest': 'saved/models/random_forest.joblib',
    'Extra Trees': 'saved/models/extra_trees.joblib', 
    'LightGBM': 'saved/models/lightgbm.joblib'
}

for name, path in model_files.items():
    try:
        models[name] = joblib.load(path)
    except Exception as e:
        print(f"Error loading {name}: {e}")

# Pydantic model for input
class PredictionInput(BaseModel):
    inputs: List[float]
    model_name: str

# Serve the main HTML page
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r") as f:
        return f.read()

# Prediction endpoint
@app.post("/predict")
async def predict(input_data: PredictionInput):
    if len(input_data.inputs) != len(features):
        raise HTTPException(status_code=400, detail="Invalid number of inputs")
    
    if input_data.model_name not in models:
        raise HTTPException(status_code=400, detail="Invalid model name")
    
    try:
        input_dict = dict(zip(features, input_data.inputs))
        input_df = pd.DataFrame([input_dict])
        
        # Compute engineered features
        engineered_df = create_engineered_features(input_df)
        
        # Handle categorical encoding if needed (assuming encoder exists if required)
        categorical_columns = engineered_df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            encoder_path = 'saved/preprocessing/encoder.joblib'
            encoder = joblib.load(encoder_path)
            engineered_df[categorical_columns] = encoder.transform(engineered_df[categorical_columns])
        
        # Predict
        model = models[input_data.model_name]
        prediction = model.predict(engineered_df)[0]
        
        # Prepare response
        original_features = {short_names[i]: input_data.inputs[i] for i in range(len(short_names))}
        engineered_features = {}
        for col in engineered_df.columns:
            if col not in features:  # Only engineered
                value = engineered_df[col].iloc[0]
                engineered_features[col] = float(value) if isinstance(value, (float, np.number)) else str(value)
        
        return {
            "prediction": float(prediction),
            "original_features": original_features,
            "engineered_features": engineered_features
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/hyperparameters/{model_name}")
async def get_hyperparameters(model_name: str):
    if model_name not in models:
        raise HTTPException(status_code=400, detail="Invalid model name")
    try:
        params = models[model_name].get_params()
        return {"model_name": model_name, "hyperparameters": params}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving hyperparameters: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)