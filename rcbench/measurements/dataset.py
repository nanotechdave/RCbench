import numpy as np
import pandas as pd
import re
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from rcbench.measurements.parser import MeasurementParser
from rcbench.logger import get_logger

logger = get_logger(__name__)

class MeasurementType(Enum):
    """Enumeration of possible measurement types."""
    NLT = "nlt"
    MEMORY_CAPACITY = "mc"
    KERNEL_RANK = "kernel"
    ACTIVATION = "activation"
    UNKNOWN = "unknown"

class ReservoirDataset:
    """
    Main class for reservoir computing measurement data.
    Handles loading, parsing, and accessing measurement data.
    """
    def __init__(self, 
                source: Union[str, pd.DataFrame, Path],
                time_column: str = 'Time[s]',
                ground_threshold: float = 1e-2,
                input_electrodes: Optional[List[str]] = None,
                ground_electrodes: Optional[List[str]] = None,
                node_electrodes: Optional[List[str]] = None):
        """
        Initialize a ReservoirDataset from a file or existing DataFrame.
        
        Args:
            source (Union[str, pd.DataFrame, Path]): Path to measurement file or DataFrame
            time_column (str): Name of the time column
            ground_threshold (float): Threshold for identifying ground electrodes
            input_electrode (Optional[List[str]]): Force specific electrodes as input
            ground_electrode (Optional[List[str]]): Force specific electrodes as ground
            node_electrodes (Optional[List[str]]): Force specific electrodes as nodes
        """
        self.time_column = time_column
        self.ground_threshold = ground_threshold
        self.measurement_type = MeasurementType.UNKNOWN
        self.forced_inputs = input_electrodes
        self.forced_grounds = ground_electrodes
        self.forced_nodes = node_electrodes
        
        # Load data from file path or use provided DataFrame
        if isinstance(source, (str, Path)):
            self.file_path = str(source)
            self.dataframe = self._load_data_from_file(source)
            self.measurement_type = self._determine_measurement_type(source)
        else:
            self.file_path = None
            self.dataframe = source
        
        # Extract electrode columns for reference
        self.voltage_columns = [col for col in self.dataframe.columns if col.endswith('_V[V]')]
        self.current_columns = [col for col in self.dataframe.columns if col.endswith('_I[A]')]
        
        # Identify electrodes using the parser
        identified_electrodes = MeasurementParser.identify_electrodes(
            self.dataframe, 
            ground_threshold=self.ground_threshold,
            forced_inputs=self.forced_inputs,
            forced_grounds=self.forced_grounds
        )
        
        # Store electrode information in this object
        self.input_electrodes = self.forced_inputs if self.forced_inputs is not None else identified_electrodes['input_electrodes']
        self.ground_electrodes = self.forced_grounds if self.forced_grounds is not None else identified_electrodes['ground_electrodes']
        self.node_electrodes = self.forced_nodes if self.forced_nodes is not None else identified_electrodes['node_electrodes']
        
        logger.info(f"Measurement type: {self.measurement_type.value if self.measurement_type else 'Unknown'}")
    
    def _load_data_from_file(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a file into a DataFrame.
        
        Args:
            file_path (str): Path to the measurement file
            
        Returns:
            pd.DataFrame: Loaded and cleaned DataFrame
        """
        # Convert Path object to string if needed
        if hasattr(file_path, 'resolve'):
            file_path = str(file_path)
            
        logger.info(f"Loading data from {file_path}")
        try:
            df = pd.read_csv(file_path, sep=r'\s+', engine='python')
            
            # Clean data
            df.replace('nan', pd.NA, inplace=True)
            df.dropna(axis=1, how='any', inplace=True)
            df = df.astype(float)
            
            return df
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def _determine_measurement_type(self, file_path: str) -> MeasurementType:
        """
        Determine the type of measurement based on the filename.
        
        Args:
            file_path (str): Path to the measurement file
            
        Returns:
            MeasurementType: Type of measurement
        """
        # Convert Path object to string if needed
        if hasattr(file_path, 'resolve'):
            file_path = str(file_path)
            
        filename = file_path.lower()
        if "nlt" in filename or "nonlinear" in filename or "sin" in filename:
            return MeasurementType.NLT
        elif "mc" in filename or "memory" in filename:
            return MeasurementType.MEMORY_CAPACITY
        elif "kernel" in filename:
            return MeasurementType.KERNEL_RANK
        elif "activation" in filename or "constant" in filename:
            return MeasurementType.ACTIVATION
        return MeasurementType.UNKNOWN
    
    @property
    def time(self) -> np.ndarray:
        """Get time data as a numpy array."""
        return self.dataframe[self.time_column].to_numpy()

    @property
    def voltage(self) -> np.ndarray:
        """Get all voltage data as a numpy array."""
        return self.dataframe[self.voltage_columns].to_numpy()

    @property
    def current(self) -> np.ndarray:
        """Get all current data as a numpy array."""
        return self.dataframe[self.current_columns].to_numpy()
    
    def get_input_voltages(self) -> Dict[str, np.ndarray]:
        """
        Get voltage data for all input electrodes.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping electrode names to voltage arrays
        """
        return MeasurementParser.get_input_voltages(self.dataframe, self.input_electrodes)

    def get_input_currents(self) -> Dict[str, np.ndarray]:
        """
        Get current data for all input electrodes.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping electrode names to current arrays
        """
        return MeasurementParser.get_input_currents(self.dataframe, self.input_electrodes)

    def get_ground_voltages(self) -> Dict[str, np.ndarray]:
        """
        Get voltage data for all ground electrodes.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping electrode names to voltage arrays
        """
        return MeasurementParser.get_ground_voltages(self.dataframe, self.ground_electrodes)

    def get_ground_currents(self) -> Dict[str, np.ndarray]:
        """
        Get current data for all ground electrodes.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping electrode names to current arrays
        """
        return MeasurementParser.get_ground_currents(self.dataframe, self.ground_electrodes)

    def get_node_voltages(self) -> np.ndarray:
        """
        Get voltage data for all node electrodes.
        
        Returns:
            np.ndarray: Matrix of node voltages [samples, electrodes]
        """
        return MeasurementParser.get_node_voltages(self.dataframe, self.node_electrodes)
    
    def get_node_voltage(self, node: str) -> np.ndarray:
        """
        Get voltage data for a specific node electrode.
        
        Args:
            node (str): Electrode name
            
        Returns:
            np.ndarray: Voltage data for the specified node
        """
        return MeasurementParser.get_node_voltage(self.dataframe, node, self.node_electrodes)
    
    def summary(self) -> Dict:
        """
        Get a summary of the dataset including electrodes.
        
        Returns:
            Dict: Summary information
        """
        return {
            'measurement_type': self.measurement_type.value,
            'input_electrodes': self.input_electrodes,
            'ground_electrodes': self.ground_electrodes,
            'node_electrodes': self.node_electrodes,
            'time_column': self.time_column,
            'voltage_columns': self.voltage_columns,
            'current_columns': self.current_columns,
            'data_shape': self.dataframe.shape
        }
