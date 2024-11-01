#%% required libraries
from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field
from typing import Literal
from catboost import CatBoostRegressor
import pandas as pd
from steps.predict import Predictor

app = FastAPI()

# Input schema for the insured person
class Insured(BaseModel):
    VehPower: int = Field(title='Vehicle Power', description='Vehicle power in CV', ge=1, le=20, default=5)
    VehAge: int = Field(title='Vehicle Age', description='Vehicle age in years', ge=0, le=120, default=1)
    DrivAge: int = Field(title='Driver Age', description='Driver age in years', ge=18, le=120, default=35)
    Density: int = Field(title='Density', description='Density of inhabitants per km2', gt=0, le=30000, default=100)
    BonusMalus: int = Field(title='Bonus Malus', description='Bonus Malus', ge=50, le=230, default=100)
    VehBrand: Literal['B12', 'B3', 'B2', 'B5', 'B4', 'B6', 'B10', 'B1', 'B13', 'B11', 'B14'] = Field(title='Vehicle Brand', description='Vehicle brand as per allowed values')
    VehGas: Literal['Regular', 'Diesel'] = Field(title='Vehicle Gas', description='Vehicle gas as per allowed values')
    Region: Literal['R72', 'R91', 'R52', 'R11', 'R94', 'R93', 'R31', 'R82', 'R22', 'R21', 'R42', 'R54', 'R73', 'R41', 'R26', 'R25', 'R24', 'R53', 'R83', 'R23', 'R74', 'R43'] = Field(title='Region', description='Region code as per allowed values')
    Area: Literal['A', 'B', 'C', 'D', 'E', 'F', 'G'] = Field(title='Area', description='Area code as per allowed values')

# Output schema for prediction response
class PredictionResponse(BaseModel):
    Frequency: float
    Severity: float
    Pure_Premium: float

# Load models as a startup event
@app.on_event("startup")
def load_models():
    global model_freq, model_sev
    model_freq = CatBoostRegressor()
    model_freq.load_model('models/frequency_model.cbm')
    model_sev = CatBoostRegressor()
    model_sev.load_model('models/severity_model.cbm')


# Dependency for loading the Predictor
def get_predictor(model, type_):
    return Predictor(model, type_)


@app.get("/", response_model=dict)
async def read_root():
    return {"Health check": "OK"}

# Prediction endpoint
@app.post("/predict/", response_model=PredictionResponse)
async def predict(
    insured: Insured,
    predictor_freq: Predictor = Depends(lambda: get_predictor(model_freq, 'frequency')),
    predictor_sev: Predictor = Depends(lambda: get_predictor(model_sev, 'severity'))
):
    # Convert insured data to DataFrame
    df_insured = pd.DataFrame([insured.dict()])
    df_insured['log_exposure'] = 0.0
    df_insured['ClaimNb'] = 1
    
    # Perform predictions
    pred_freq = predictor_freq.predict(df_insured)[0]
    pred_sev = predictor_sev.predict(df_insured)[0]
    pure_premium = pred_freq * pred_sev
    
    # Create response
    return PredictionResponse(Frequency=pred_freq, Severity=pred_sev, Pure_Premium=pure_premium)