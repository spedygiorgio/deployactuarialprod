#%% Importing necessary libraries
"""Modulo per la creazione dei pool di previsione e la predizione con CatBoost."""
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from typing import Literal

from utils import get_logger, timer

logger = get_logger(__name__)

#%% predictor class
class Predictor:
    """Wrapper che prepara i dati per la predizione e invoca il modello CatBoost."""
    def __init__(self,  model: CatBoostRegressor, model_type=Literal['frequency','severity']):
        self.model = model
        self.model_type = model_type
        self.cat_features = ['VehBrand', 'VehGas', 'Region','Area']
        self.numeric_features = ['VehPower', 'VehAge', 'DrivAge', 'Density', 'BonusMalus']
    
    @timer
    def create_pool(self, data: pd.DataFrame | dict) -> Pool:
        """Costruisce un `Pool` da un DataFrame o da un dizionario singolo.

        Se viene fornito un dict, viene creato un DataFrame ad una riga e vengono aggiunte
        colonne fittizie (`Exposure`, `log_exposure`, `ClaimNb`) se necessarie.
        """
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
        logger.info(f"Pool creato per predizione (nrows={len(data)}) - tipo={self.model_type}")
        return pool
    
    @timer
    def predict(self, data: pd.DataFrame) -> np.array:
        """Esegue la predizione usando il modello fornito e ritorna un array numpy."""
        pool = self.create_pool(data)
        preds = self.model.predict(pool)
        logger.info(f"Predizioni calcolate (n={len(preds)})")
        return preds