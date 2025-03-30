import pandas as pd
import numpy as np
from typing import List
from rcbench.measurements.dataset import ReservoirDataset
from rcbench.logger import get_logger

logger = get_logger(__name__)
class MeasurementLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.dataframe: pd.DataFrame = None
        self.voltage_columns: List = []
        self.current_columns: List = []
        self.time_column: str = 'Time[s]'

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from a whitespace-separated file into a Pandas DataFrame.
        Automatically identifies voltage and current columns.
        """
        self.dataframe = pd.read_csv(self.file_path, sep='\s+', engine='python')
        self._identify_columns()
        self._clean_data()
        return self.dataframe

    def _identify_columns(self):
        """
        Automatically identifies voltage and current columns based on naming conventions.
        """
        if self.dataframe is not None:
            self.voltage_columns = [col for col in self.dataframe.columns if '_V[V]' in col]
            self.current_columns = [col for col in self.dataframe.columns if '_I[A]' in col]
        else:
            raise ValueError("Dataframe is not loaded. Call load_data() first.")

    def _clean_data(self):
        """
        Cleans the data by removing columns containing NaNs or replacing them if necessary.
        """
        self.dataframe.replace('nan', pd.NA, inplace=True)
        self.dataframe.dropna(axis=1, how='any', inplace=True)
        self.dataframe = self.dataframe.astype(float)

    def get_voltage_data(self) -> np.ndarray:
        """
        Returns voltage data as a numpy array.
        """
        return self.dataframe[self.voltage_columns].to_numpy()
    
    def get_dataset(self) -> ReservoirDataset:
        """
        Returns a ReservoirDataset instance directly.
        """
        if self.dataframe is None:
            self.load_data()
        return ReservoirDataset(
            dataframe=self.dataframe,
            time_column=self.time_column,
            voltage_columns=self.voltage_columns,
            current_columns=self.current_columns
        )

    def get_current_data(self) -> np.ndarray:
        """
        Returns current data as a numpy array.
        """
        return self.dataframe[self.current_columns].to_numpy()

    def get_time_data(self) -> np.ndarray:
        """
        Returns the time data as a numpy array.
        """
        return self.dataframe[self.time_column].to_numpy()
    