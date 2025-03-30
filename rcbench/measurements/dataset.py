import numpy as np
import pandas as pd
from typing import Dict, List
from rcbench.logger import get_logger

logger = get_logger(__name__)
class ReservoirDataset:
    def __init__(self, 
                 dataframe: pd.DataFrame, 
                 time_column: str, 
                 voltage_columns: List[str] , 
                 current_columns: List[str]
                 ) -> None:
        self.dataframe: pd.DataFrame = dataframe
        self.time_column: str = time_column
        self.voltage_columns: List[str] = voltage_columns
        self.current_columns: List[str] = current_columns

    @property
    def time(self) -> np.ndarray:
        return self.dataframe[self.time_column].to_numpy()

    @property
    def voltage(self) -> np.ndarray:
        return self.dataframe[self.voltage_columns].to_numpy()

    @property
    def current(self) -> np.ndarray:
        return self.dataframe[self.current_columns].to_numpy()

    def summary(self) -> Dict:
        return {
            'time_column': self.time_column,
            'voltage_columns': self.voltage_columns,
            'current_columns': self.current_columns,
            'data_shape': self.dataframe.shape
        }
