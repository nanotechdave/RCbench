import numpy as np
import pandas as pd
from rcbench.logger import get_logger

logger = get_logger(__name__)
class ReservoirDataset:
    def __init__(self, dataframe, time_column, voltage_columns, current_columns):
        self.dataframe = dataframe
        self.time_column = time_column
        self.voltage_columns = voltage_columns
        self.current_columns = current_columns

    @property
    def time(self):
        return self.dataframe[self.time_column].to_numpy()

    @property
    def voltage(self):
        return self.dataframe[self.voltage_columns].to_numpy()

    @property
    def current(self):
        return self.dataframe[self.current_columns].to_numpy()

    def summary(self):
        return {
            'time_column': self.time_column,
            'voltage_columns': self.voltage_columns,
            'current_columns': self.current_columns,
            'data_shape': self.dataframe.shape
        }
