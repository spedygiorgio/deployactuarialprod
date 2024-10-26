#%% import libraries
import pandas as pd
import numpy as np
from sklearn.metrics import mean_poisson_deviance, root_mean_squared_error
from catboost import CatBoostRegressor, Pool
from catboost.utils import get_gpu_device_count
#%% trainer class
class Trainer:
    def __init__(self, train_data: pd.DataFrame, val_data: pd.DataFrame, test_data:pd.DataFrame, target: str, model_type='frequency'):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.target = target
        self.model_type = model_type
        self.cat_features = ['VehBrand', 'VehGas', 'Region','Area']
        self.numeric_features = ['VehPower', 'VehAge', 'DrivAge', 'Density', 'BonusMalus']
        # check GPU availability
        self.gpu_available = get_gpu_device_count() > 0
    
    def _ap_ratio(self, actual, predicted) -> float: 
        return np.sum(actual) / np.sum(predicted)


    def create_pools(self) -> tuple[Pool, Pool, Pool]:
        features = self.numeric_features + self.cat_features
        if self.model_type == 'frequency':
            train_pool = Pool(data=self.train_data.filter(items=features), 
                            label=self.train_data[self.target], 
                            cat_features=self.cat_features, baseline=self.train_data['log_exposure'])
            val_pool = Pool(data=self.val_data.filter(items=features), 
                            label=self.val_data[self.target], 
                            cat_features=self.cat_features, baseline=self.val_data['log_exposure'])
            test_pool = Pool(data=self.test_data.filter(items=features), 
                            label=self.test_data[self.target], 
                            cat_features=self.cat_features, baseline=self.test_data['log_exposure'])
        else:
            train_pool = Pool(data=self.train_data.filter(items=features), 
                            label=self.train_data[self.target], 
                            cat_features=self.cat_features, weight=self.train_data['ClaimNb'])
            val_pool = Pool(data=self.val_data.filter(items=features), 
                            label=self.val_data[self.target], 
                            cat_features=self.cat_features, weight=self.val_data['ClaimNb'])
            test_pool = Pool(data=self.test_data.filter(items=features), 
                            label=self.test_data[self.target], 
                            cat_features=self.cat_features, weight=self.test_data['ClaimNb'])
        return train_pool, val_pool, test_pool

    def train_model(self) -> CatBoostRegressor:
        train_pool, val_pool, _ = self.create_pools()
        model_task = 'GPU' if self.gpu_available else 'CPU'
        model = CatBoostRegressor(loss_function='Poisson' if self.model_type == 'frequency' else 'RMSE', task_type=model_task, iterations=2000)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=100)  
        return model
    
    def evaluate_model(self, model: CatBoostRegressor, pool: Pool) -> tuple[float, float]:
        predicted_values = model.predict(pool)
        actual_values = pool.get_label()

        if self.model_type == 'frequency':
            ap_ratio_to_return = self._ap_ratio(actual=actual_values, predicted=predicted_values)
            out = ap_ratio_to_return, mean_poisson_deviance(y_true=actual_values, y_pred=predicted_values)
            return out
        else:
            mask = np.array(pool.get_weight()) > 0
            actual_values = actual_values[mask]
            predicted_values = predicted_values[mask]
            ap_ratio_to_return = self._ap_ratio(actual=actual_values, predicted=predicted_values)
            out = ap_ratio_to_return, root_mean_squared_error(y_true=actual_values, y_pred=predicted_values)
            return out
    
    def save_model(self, model: CatBoostRegressor, path: str):
        model.save_model(path)
        return None
    