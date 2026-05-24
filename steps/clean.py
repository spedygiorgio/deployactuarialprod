#%% load libraries
"""Utilities per pulizia e preprocessing minimale dei dataset."""
import pandas as pd
import numpy as np
from typing import Literal

from utils import get_logger, timer

logger = get_logger(__name__)


class Cleaner:
    """Classe che applica semplici regole di pulizia ai DataFrame.

    Attualmente rimuove righe con NA e, per la frequenza, righe con Exposure==0.
    """
    def __init__(self, data_folder_path="./data"):
        self.data_folder_path = data_folder_path

    @timer
    def clean(self, df: pd.DataFrame, type: Literal['frequency','severity']) -> pd.DataFrame:
        """Pulisce il DataFrame in base al tipo di modello.

        Args:
            df: DataFrame da pulire
            type: 'frequency' o 'severity' per regole specifiche

        Returns:
            DataFrame pulito
        """
        before_rows = len(df)
        df = df.dropna()
        if type == "frequency":
            df = df[df["Exposure"] != 0]
        after_rows = len(df)
        logger.info(f"Pulizia completata ({type}): righe {before_rows} -> {after_rows})")
        return df