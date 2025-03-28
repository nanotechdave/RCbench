import numpy as np
import pandas as pd
import re
from rcda.logger import get_logger

logger = get_logger(__name__)
class MeasurementParser:
    
    def __init__(self, dataset, ground_threshold=1e-2):
        """
        Parses measurement data to identify input electrodes (voltage signals
        associated with non-nan current measurements) and ground electrodes 
        (non-nan current, voltage steadily close to zero).
        """
        self.dataset = dataset
        self.dataframe = dataset.dataframe
        self.time = dataset.time

        # Extract only the columns that still exist after cleaning
        self.voltage_cols = [col for col in self.dataframe.columns if col.endswith('_V[V]')]
        self.current_cols = [col for col in self.dataframe.columns if col.endswith('_I[A]')]

        self.input_electrodes, self.ground_electrodes = self._find_input_and_ground(ground_threshold)
        self.node_electrodes = self._identify_nodes()
        logger.info(f"Input electrodes: {self.input_electrodes}")
        logger.info(f"Ground electrodes: {self.ground_electrodes}")
        logger.info(f"Node electrodes: {self.node_electrodes}")
        logger.info(f"Total node voltages: {len(self.node_electrodes)}")


    def _find_input_and_ground(self, ground_threshold):
        input_electrodes = []
        ground_electrodes = []

        for current_col in self.current_cols:
            electrode = current_col.split('_')[0]
            voltage_col = f"{electrode}_V[V]"

            if voltage_col in self.voltage_cols:
                voltage_data = self.dataframe[voltage_col].values

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
            raise ValueError("No input electrodes found.")
        if not ground_electrodes:
            raise ValueError("No ground electrodes found.")

        return input_electrodes, ground_electrodes

    def _identify_nodes(self):
        exclude = set(self.input_electrodes + self.ground_electrodes)
        node_electrodes = []

        for col in self.voltage_cols:
            match = re.match(r"^(\d+)_V\[V\]$", col)
            if match:
                electrode = match.group(1)
                if electrode not in exclude:
                    node_electrodes.append(electrode)

        return list(set(node_electrodes))

    def get_input_voltages(self):
        return {elec: self.dataframe[f'{elec}_V[V]'].values for elec in self.input_electrodes}

    def get_input_currents(self):
        return {elec: self.dataframe[f'{elec}_I[A]'].values for elec in self.input_electrodes}

    def get_ground_voltages(self):
        return {elec: self.dataframe[f'{elec}_V[V]'].values for elec in self.ground_electrodes}

    def get_ground_currents(self):
        return {elec: self.dataframe[f'{elec}_I[A]'].values for elec in self.ground_electrodes}

    def get_node_voltages(self):
        cols = [f'{elec}_V[V]' for elec in self.node_electrodes]
        return self.dataframe[cols].values

    def summary(self):
        return {
            'input_electrodes': self.input_electrodes,
            'ground_electrodes': self.ground_electrodes,
            'node_electrodes': self.node_electrodes
        }
