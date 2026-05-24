#%% import key libraries
"""Module per l'ingestione dei dataset (train/valid/test)."""
import pandas as pd
from pandas.core.frame import DataFrame
import os
from typing import Tuple
import numpy as np

# utils: logger e timer
from utils import get_logger, timer

logger = get_logger(__name__)

#%% Ingestion class
class Ingestion:
    """Classe responsabile del caricamento dei file di dati.

    Estrapola trasformazioni minime (es. log_exposure, severity) e restituisce
    tuple (train, valid, test) come pandas DataFrame.
    """
    def __init__(self):
        self.data_dir = "./data"

    def _load_data(self, file_name: str, assign_func) -> DataFrame:
        """Carica un singolo file CSV e applica le assegnazioni/passaggi richiesti.

        Args:
            file_name: nome del file CSV nella cartella dati
            assign_func: dict usato da pandas `.assign` per aggiungere colonne

        Returns:
            DataFrame caricato e trasformato
        """
        file_path = os.path.join(self.data_dir, file_name)
        logger.info(f"Caricamento file: {file_path}")
        return pd.read_csv(file_path, sep=";").assign(**assign_func)

    @timer
    def load_freq(self) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """Carica i dataset per il modello di frequenza e aggiunge `log_exposure`."""
        assign_func = {'log_exposure': lambda x: np.log(x['Exposure'])}
        df_train = self._load_data('train.csv', assign_func)
        df_valid = self._load_data('valid.csv', assign_func)
        df_test = self._load_data('test.csv', assign_func)
        return df_train, df_valid, df_test

    @timer
    def load_severity(self) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """Carica i dataset per il modello di severity e calcola la severità per riga."""
        assign_func = {'severity': lambda x: x.apply(lambda row: row['claims_cost'] / row['ClaimNb'] if row['ClaimNb'] > 0 else 0, axis=1)}
        df_train = self._load_data('train.csv', assign_func)
        df_valid = self._load_data('valid.csv', assign_func)
        df_test = self._load_data('test.csv', assign_func)
        return df_train, df_valid, df_test
