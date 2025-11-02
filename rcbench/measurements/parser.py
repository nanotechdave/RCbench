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
    Utility class to parse measurement data and identify nodes.
    This class only parses data and does not store node information.
    """
    
    @staticmethod
    def identify_nodes(dataframe: pd.DataFrame, ground_threshold: float = 1e-2, 
                           forced_inputs: List[str] = None, forced_grounds: List[str] = None) -> Dict[str, List[str]]:
        """
        Parse measurement data to identify input, ground, and computation nodes.
        
        Args:
            dataframe: DataFrame containing measurement data
            ground_threshold: Threshold for identifying ground nodes
            forced_inputs: Optional list of nodes to force as input
            forced_grounds: Optional list of nodes to force as ground
            
        Returns:
            Dictionary containing identified input, ground, and computation nodes
        """
        # Extract columns
        voltage_cols = [col for col in dataframe.columns if col.endswith('_V[V]')]
        current_cols = [col for col in dataframe.columns if col.endswith('_I[A]')]
        
        # Use forced nodes if provided
        if forced_inputs is not None and forced_grounds is not None:
            input_nodes = forced_inputs
            ground_nodes = forced_grounds
        else:
            # Find input and ground nodes
            input_nodes, ground_nodes = MeasurementParser._find_input_and_ground(
                dataframe, voltage_cols, current_cols, ground_threshold
            )
        
        # Identify computation nodes
        nodes = MeasurementParser._identify_computation_nodes(
            voltage_cols, input_nodes, ground_nodes
        )
        
        logger.info(f"Identified input nodes: {input_nodes}")
        logger.info(f"Identified ground nodes: {ground_nodes}")
        logger.info(f"Identified computation nodes: {nodes}")
        logger.info(f"Total node voltages: {len(nodes)}")
        
        return {
            'input_nodes': input_nodes,
            'ground_nodes': ground_nodes,
            'nodes': nodes
        }

    @staticmethod
    def _find_input_and_ground(dataframe: pd.DataFrame, voltage_cols: List[str], 
                              current_cols: List[str], ground_threshold: float) -> Tuple[List[str], List[str]]:
        """
        Identify input and ground nodes based on voltage and current measurements.
        
        Args:
            dataframe: DataFrame containing measurement data
            voltage_cols: List of voltage column names
            current_cols: List of current column names
            ground_threshold: Threshold for identifying ground nodes
            
        Returns:
            Tuple of (input_nodes, ground_nodes)
        """
        input_nodes = []
        ground_nodes = []

        for current_col in current_cols:
            # Extract node name from current column (remove "_I[A]" suffix)
            if current_col.endswith('_I[A]'):
                node = current_col[:-5]  # Remove "_I[A]" suffix
                voltage_col = f"{node}_V[V]"

                if voltage_col in voltage_cols:
                    voltage_data = dataframe[voltage_col].values

                    # Skip if all values are NaN (these columns should have been dropped already)
                    if np.all(np.isnan(voltage_data)):
                        logger.info(f"Skipping node {node} - all voltage values are NaN")
                        continue

                    # Check if the voltage is close to 0 (low std & low mean) - indicates ground
                    is_ground = (
                        np.nanstd(voltage_data) < ground_threshold and
                        np.abs(np.nanmean(voltage_data)) < ground_threshold
                    )

                    # Simple classification based on voltage characteristics:
                    # - If voltage is consistently near zero -> ground node
                    # - Otherwise -> input node (nodes that drive the system)
                    if is_ground:
                        ground_nodes.append(node)
                    else:
                        input_nodes.append(node)

        if not input_nodes:
            logger.warning("No input nodes found.")
        if not ground_nodes:
            logger.warning("No ground nodes found.")

        return input_nodes, ground_nodes

    @staticmethod
    def _identify_computation_nodes(voltage_cols: List[str], input_nodes: List[str], 
                       ground_nodes: List[str]) -> List[str]:
        """
        Identify computation nodes (nodes that are neither input nor ground).
        
        Args:
            voltage_cols: List of voltage column names
            input_nodes: List of input node names
            ground_nodes: List of ground node names
            
        Returns:
            List of computation node names
        """
        exclude = set(input_nodes + ground_nodes)
        nodes = []

        for col in voltage_cols:
            # Extract node name from voltage column (remove "_V[V]" suffix)
            if col.endswith('_V[V]'):
                node = col[:-5]  # Remove "_V[V]" suffix
                
                # If this node is not in the input or ground nodes, it's a computation node
                if node not in exclude:
                    nodes.append(node)

        # Sort nodes alphabetically (since names can be arbitrary)
        return sorted(list(set(nodes)))

    @staticmethod
    def get_input_voltages(dataframe: pd.DataFrame, input_nodes: List[str]) -> Dict[str, np.ndarray]:
        """
        Get voltage data for input nodes.
        
        Args:
            dataframe: DataFrame containing measurement data
            input_nodes: List of input node names
            
        Returns:
            Dictionary mapping node names to voltage arrays
        """
        return {node: dataframe[f'{node}_V[V]'].values for node in input_nodes}

    @staticmethod
    def get_input_currents(dataframe: pd.DataFrame, input_nodes: List[str]) -> Dict[str, np.ndarray]:
        """
        Get current data for input nodes.
        
        Args:
            dataframe: DataFrame containing measurement data
            input_nodes: List of input node names
            
        Returns:
            Dictionary mapping node names to current arrays
        """
        return {node: dataframe[f'{node}_I[A]'].values for node in input_nodes}

    @staticmethod
    def get_ground_voltages(dataframe: pd.DataFrame, ground_nodes: List[str]) -> Dict[str, np.ndarray]:
        """
        Get voltage data for ground nodes.
        
        Args:
            dataframe: DataFrame containing measurement data
            ground_nodes: List of ground node names
            
        Returns:
            Dictionary mapping node names to voltage arrays
        """
        return {node: dataframe[f'{node}_V[V]'].values for node in ground_nodes}

    @staticmethod
    def get_ground_currents(dataframe: pd.DataFrame, ground_nodes: List[str]) -> Dict[str, np.ndarray]:
        """
        Get current data for ground nodes.
        
        Args:
            dataframe: DataFrame containing measurement data
            ground_nodes: List of ground node names
            
        Returns:
            Dictionary mapping node names to current arrays
        """
        return {node: dataframe[f'{node}_I[A]'].values for node in ground_nodes}

    @staticmethod
    def get_node_voltages(dataframe: pd.DataFrame, nodes: List[str]) -> np.ndarray:
        """
        Get voltage data for all computation nodes.
        
        Args:
            dataframe: DataFrame containing measurement data
            nodes: List of computation node names
            
        Returns:
            Matrix of node voltages [samples, nodes]
        """
        cols = [f'{node}_V[V]' for node in nodes]
        return dataframe[cols].values
    
    @staticmethod
    def get_node_voltage(dataframe: pd.DataFrame, node: str, nodes: List[str]) -> np.ndarray:
        """
        Get voltage data for a specific computation node.
        
        Args:
            dataframe: DataFrame containing measurement data
            node: Node name
            nodes: List of computation node names
            
        Returns:
            Voltage data for the specified node
        """
        if node in nodes:
            return dataframe[f'{node}_V[V]'].values
        raise ValueError(f"Computation node '{node}' not found.")

    def summary(self, identified_nodes: Dict[str, List[str]]) -> Dict:
        return {
            'input_nodes': identified_nodes['input_nodes'],
            'ground_nodes': identified_nodes['ground_nodes'],
            'nodes': identified_nodes['nodes']
        }
