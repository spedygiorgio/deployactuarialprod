#%% load libraries
import pandas as pd
import numpy as np
from typing import Literal

class Cleaner:
    def __init__(self, data_folder_path="./data"):
        self.data_folder_path = data_folder_path

    def clean(self, df: pd.DataFrame, type: Literal['frequency','severity']) -> pd.DataFrame:
        df = df.dropna()
        if type == "frequency":
            df = df[df["Exposure"] != 0]
        return df