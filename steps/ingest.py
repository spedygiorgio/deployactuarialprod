#%% import key libraries
import pandas as pd
from pandas.core.frame import DataFrame
import os
from typing import Tuple
import numpy as np

#%% Ingestion class
class Ingestion:
    def __init__(self):
        self.data_dir = "./data"

    def _load_data(self, file_name: str, assign_func) -> DataFrame:
        file_path = os.path.join(self.data_dir, file_name)
        return pd.read_csv(file_path, sep=";").assign(**assign_func)

    def load_freq(self) -> Tuple[DataFrame, DataFrame, DataFrame]:
        assign_func = {'log_exposure': lambda x: np.log(x['Exposure'])}
        df_train = self._load_data('train.csv', assign_func)
        df_valid = self._load_data('valid.csv', assign_func)
        df_test = self._load_data('test.csv', assign_func)
        return df_train, df_valid, df_test

    def load_severity(self) -> Tuple[DataFrame, DataFrame, DataFrame]:
        assign_func = {'severity': lambda x: x.apply(lambda row: row['claims_cost'] / row['ClaimNb'] if row['ClaimNb'] > 0 else 0, axis=1)}
        df_train = self._load_data('train.csv', assign_func)
        df_valid = self._load_data('valid.csv', assign_func)
        df_test = self._load_data('test.csv', assign_func)
        return df_train, df_valid, df_test
