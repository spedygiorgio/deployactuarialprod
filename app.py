#%% required libraries
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
from catboost import CatBoostRegressor
import pandas as pd
from steps.predict import Predictor

app = FastAPI()

# define the input schema
class Insured(BaseModel):
    VehPower: int = Field(title='Vehicle Power', description='Vehicle power in CV', ge=1, le=20)
    VehAge: int = Field(title='Vehicle Age', description='Vehicle age in years', ge=0, le=120)
    DrivAge: int = Field(title='Driver Age', description='Driver age in years', ge=18, le=120)
    Density: int = Field(title='Density', description='Density of inhabitants per km2', gt=0, le=30000)
    BonusMalus: int = Field(title='Bonus Malus', description='Bonus Malus', ge=50, le=230)
    VehBrand: Literal['B12', 'B3', 'B2', 'B5', 'B4', 'B6', 'B10', 'B1', 'B13', 'B11', 'B14'] = Field(title='Vehicle Brand', description='Vehicle brand as per allowed values')
    VehGas: Literal['Regular', 'Diesel'] = Field(title='Vehicle Gas', description='Vehicle gas as per allowed values')
    Region: Literal['R72', 'R91', 'R52', 'R11', 'R94', 'R93', 'R31', 'R82', 'R22', 'R21', 'R42','R54', 'R73', 'R41', 'R26', 'R25', 'R24', 'R53', 'R83', 'R23', 'R74', 'R43'] = Field(title='Region', description='Region code as per allowed values')
    Area: Literal['A', 'B', 'C', 'D', 'E', 'F', 'G'] = Field(title='Area', description='Area code as per allowed values')

model_freq = CatBoostRegressor().load_model('models/frequency_model.cbm')
model_sev: CatBoostRegressor = CatBoostRegressor().load_model('models/severity_model.cbm')


@app.get("/")
async def read_root():
    return {"Health check": "OK"}

@app.post("/predict/")
async def predict(insured: Insured):
    df_insured = pd.DataFrame([insured.dict()])
    # adds the log exposure = 0 and fake ClaimNb = 1
    df_insured['log_exposure'] = 0.0
    df_insured['ClaimNb'] = 1
    pred_freq = Predictor(model_freq, 'frequency').predict(df_insured)
    pred_sev = Predictor(model_sev, 'severity').predict(df_insured)
    pred_pp = pred_freq[0] * pred_sev[0]
    item = {
        "Frequency": pred_freq[0],
        "Severity": pred_sev[0],
        "Pure Premium": pred_pp
    }
    return item