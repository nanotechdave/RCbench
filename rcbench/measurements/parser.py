import numpy as np
import pandas as pd
import re
from typing import List, Tuple, Dict, Any, TYPE_CHECKING
from rcbench.logger import get_logger

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from rcbench.measurements.dataset import ReservoirDataset

logger = get_logger(__name__)

class MeasurementParser:
    """
    Utility class to parse measurement data and identify electrodes.
    This class only parses data and does not store electrode information.
    """
    
    @staticmethod
    def identify_electrodes(dataframe: pd.DataFrame, ground_threshold: float = 1e-2, 
                           forced_inputs: List[str] = None, forced_grounds: List[str] = None) -> Dict[str, List[str]]:
        """
        Parse measurement data to identify input, ground, and node electrodes.
        
        Args:
            dataframe: DataFrame containing measurement data
            ground_threshold: Threshold for identifying ground electrodes
            forced_inputs: Optional list of electrodes to force as input
            forced_grounds: Optional list of electrodes to force as ground
            
        Returns:
            Dictionary containing identified input, ground, and node electrodes
        """
        # Extract columns
        voltage_cols = [col for col in dataframe.columns if col.endswith('_V[V]')]
        current_cols = [col for col in dataframe.columns if col.endswith('_I[A]')]
        
        # Use forced electrodes if provided
        if forced_inputs is not None and forced_grounds is not None:
            input_electrodes = forced_inputs
            ground_electrodes = forced_grounds
        else:
            # Find input and ground electrodes
            input_electrodes, ground_electrodes = MeasurementParser._find_input_and_ground(
                dataframe, voltage_cols, current_cols, ground_threshold
            )
        
        # Identify node electrodes
        node_electrodes = MeasurementParser._identify_nodes(
            voltage_cols, input_electrodes, ground_electrodes
        )
        
        logger.info(f"Identified input electrodes: {input_electrodes}")
        logger.info(f"Identified ground electrodes: {ground_electrodes}")
        logger.info(f"Identified node electrodes: {node_electrodes}")
        logger.info(f"Total node voltages: {len(node_electrodes)}")
        
        return {
            'input_electrodes': input_electrodes,
            'ground_electrodes': ground_electrodes,
            'node_electrodes': node_electrodes
        }

    @staticmethod
    def _find_input_and_ground(dataframe: pd.DataFrame, voltage_cols: List[str], 
                              current_cols: List[str], ground_threshold: float) -> Tuple[List[str], List[str]]:
        """
        Identify input and ground electrodes based on voltage and current measurements.
        
        Args:
            dataframe: DataFrame containing measurement data
            voltage_cols: List of voltage column names
            current_cols: List of current column names
            ground_threshold: Threshold for identifying ground electrodes
            
        Returns:
            Tuple of (input_electrodes, ground_electrodes)
        """
        input_electrodes = []
        ground_electrodes = []

        for current_col in current_cols:
            electrode = current_col.split('_')[0]
            voltage_col = f"{electrode}_V[V]"

            if voltage_col in voltage_cols:
                voltage_data = dataframe[voltage_col].values

                # Check if the voltage is close to 0 (low std & low mean)
                is_ground = (
                    np.nanstd(voltage_data) < ground_threshold and
                    np.abs(np.nanmean(voltage_data)) < ground_threshold
                )

                if is_ground:
                    ground_electrodes.append(electrode)
                else:
                    input_electrodes.append(electrode)

        if not input_electrodes:
            logger.warning("No input electrodes found.")
        if not ground_electrodes:
            logger.warning("No ground electrodes found.")

        return input_electrodes, ground_electrodes

    @staticmethod
    def _identify_nodes(voltage_cols: List[str], input_electrodes: List[str], 
                       ground_electrodes: List[str]) -> List[str]:
        """
        Identify node electrodes (electrodes that are neither input nor ground).
        
        Args:
            voltage_cols: List of voltage column names
            input_electrodes: List of input electrode names
            ground_electrodes: List of ground electrode names
            
        Returns:
            List of node electrode names
        """
        exclude = set(input_electrodes + ground_electrodes)
        node_electrodes = []

        for col in voltage_cols:
            match = re.match(r"^(\d+)_V\[V\]$", col)
            if match:
                electrode = match.group(1)
                if electrode not in exclude:
                    node_electrodes.append(electrode)

        # Sort numerically by converting to int, then back to string
        return sorted(list(set(node_electrodes)), key=lambda x: int(x))

    @staticmethod
    def get_input_voltages(dataframe: pd.DataFrame, input_electrodes: List[str]) -> Dict[str, np.ndarray]:
        """
        Get voltage data for input electrodes.
        
        Args:
            dataframe: DataFrame containing measurement data
            input_electrodes: List of input electrode names
            
        Returns:
            Dictionary mapping electrode names to voltage arrays
        """
        return {elec: dataframe[f'{elec}_V[V]'].values for elec in input_electrodes}

    @staticmethod
    def get_input_currents(dataframe: pd.DataFrame, input_electrodes: List[str]) -> Dict[str, np.ndarray]:
        """
        Get current data for input electrodes.
        
        Args:
            dataframe: DataFrame containing measurement data
            input_electrodes: List of input electrode names
            
        Returns:
            Dictionary mapping electrode names to current arrays
        """
        return {elec: dataframe[f'{elec}_I[A]'].values for elec in input_electrodes}

    @staticmethod
    def get_ground_voltages(dataframe: pd.DataFrame, ground_electrodes: List[str]) -> Dict[str, np.ndarray]:
        """
        Get voltage data for ground electrodes.
        
        Args:
            dataframe: DataFrame containing measurement data
            ground_electrodes: List of ground electrode names
            
        Returns:
            Dictionary mapping electrode names to voltage arrays
        """
        return {elec: dataframe[f'{elec}_V[V]'].values for elec in ground_electrodes}

    @staticmethod
    def get_ground_currents(dataframe: pd.DataFrame, ground_electrodes: List[str]) -> Dict[str, np.ndarray]:
        """
        Get current data for ground electrodes.
        
        Args:
            dataframe: DataFrame containing measurement data
            ground_electrodes: List of ground electrode names
            
        Returns:
            Dictionary mapping electrode names to current arrays
        """
        return {elec: dataframe[f'{elec}_I[A]'].values for elec in ground_electrodes}

    @staticmethod
    def get_node_voltages(dataframe: pd.DataFrame, node_electrodes: List[str]) -> np.ndarray:
        """
        Get voltage data for all node electrodes.
        
        Args:
            dataframe: DataFrame containing measurement data
            node_electrodes: List of node electrode names
            
        Returns:
            Matrix of node voltages [samples, electrodes]
        """
        cols = [f'{elec}_V[V]' for elec in node_electrodes]
        return dataframe[cols].values
    
    @staticmethod
    def get_node_voltage(dataframe: pd.DataFrame, node: str, node_electrodes: List[str]) -> np.ndarray:
        """
        Get voltage data for a specific node electrode.
        
        Args:
            dataframe: DataFrame containing measurement data
            node: Electrode name
            node_electrodes: List of node electrode names
            
        Returns:
            Voltage data for the specified node
        """
        if node in node_electrodes:
            return dataframe[f'{node}_V[V]'].values
        raise ValueError(f"Node electrode '{node}' not found.")

    def summary(self, identified_electrodes: Dict[str, List[str]]) -> Dict:
        return {
            'input_electrodes': identified_electrodes['input_electrodes'],
            'ground_electrodes': identified_electrodes['ground_electrodes'],
            'node_electrodes': identified_electrodes['node_electrodes']
        }
