import pandas as pd
import numpy as np
from typing import List
from rcbench.measurements.dataset import ElecResDataset
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
        Loads data from a file into a Pandas DataFrame.
        Automatically identifies voltage and current columns.
        Supports both comma-separated and whitespace-separated formats.
        """
        # Try to detect the separator automatically
        # Read first line to detect format
        with open(self.file_path, 'r') as f:
            first_line = f.readline().strip()
        
        # Check if it's comma-separated (has commas and reasonable number of columns)
        comma_count = first_line.count(',')
        space_count = len(first_line.split()) - 1  # -1 because split() counts fields, not separators
        
        if comma_count > space_count and comma_count > 2:
            # Likely CSV format
            self.dataframe = pd.read_csv(self.file_path, sep=',', engine='python')
            logger.info("Detected comma-separated format")
        else:
            # Likely whitespace-separated format
            self.dataframe = pd.read_csv(self.file_path, sep=r'\s+', engine='python')
            logger.info("Detected whitespace-separated format")
            
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
        Cleans the data by handling empty values and converting to numeric format.
        Empty columns are dropped after conversion.
        """
        # Replace various representations of missing values with NaN
        self.dataframe.replace(['nan', 'NaN', 'NAN', '', ' '], np.nan, inplace=True)
        
        # Convert to numeric, coercing errors to NaN (handles empty strings)
        numeric_columns = []
        for col in self.dataframe.columns:
            try:
                self.dataframe[col] = pd.to_numeric(self.dataframe[col], errors='coerce')
                numeric_columns.append(col)
            except:
                logger.warning(f"Could not convert column {col} to numeric")
        
        # Keep only numeric columns
        self.dataframe = self.dataframe[numeric_columns]
        
        # Drop columns that are completely empty or contain only NaN values
        initial_columns = len(self.dataframe.columns)
        self.dataframe = self.dataframe.dropna(axis=1, how='all')  # Drop columns with all NaN values
        dropped_columns = initial_columns - len(self.dataframe.columns)
        
        if dropped_columns > 0:
            logger.info(f"Dropped {dropped_columns} empty columns")
        
        # Fill any remaining NaN values explicitly
        self.dataframe = self.dataframe.fillna(np.nan)

    def get_voltage_data(self) -> np.ndarray:
        """
        Returns voltage data as a numpy array.
        """
        return self.dataframe[self.voltage_columns].to_numpy()
    
    def get_dataset(self) -> ElecResDataset:
        """
        Returns an ElecResDataset instance directly.
        """
        if self.dataframe is None:
            self.load_data()
        return ElecResDataset(
            source=self.dataframe,
            time_column=self.time_column
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
    