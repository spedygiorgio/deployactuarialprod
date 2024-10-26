#%% Importing necessary libraries
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from typing import Literal

#%% predictor class
class Predictor:
    def __init__(self,  model: CatBoostRegressor, model_type=Literal['frequency','severity']):
        self.model = model
        self.model_type = model_type
        self.cat_features = ['VehBrand', 'VehGas', 'Region','Area']
        self.numeric_features = ['VehPower', 'VehAge', 'DrivAge', 'Density', 'BonusMalus']
    
    def create_pool(self, data: pd.DataFrame | dict) -> Pool:
        if isinstance(data, dict):
            data = pd.DataFrame([data])
            # add fictitious exposure column if it does not exist
            if 'Exposure' not in data.columns and self.model_type == 'frequency':
                data['Exposure'] = 1
                data = data.assign(log_exposure=np.log(data['Exposure']))
            if self.model_type == 'severity' and 'ClaimNb' not in data.columns:
                data = data.assign(ClaimNb=1)
            
        
        features = self.numeric_features + self.cat_features
        if self.model_type == 'frequency':
            pool = Pool(data=data.filter(items=features), 
                        cat_features=self.cat_features, baseline=data['log_exposure'])
        else:
            pool = Pool(data=data.filter(items=features), 
                        cat_features=self.cat_features, weight=data['ClaimNb'])
        return pool
    
    def predict(self, data: pd.DataFrame) -> np.array:
        pool = self.create_pool(data)
        return self.model.predict(pool)