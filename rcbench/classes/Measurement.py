from enum import Enum
from typing import Optional
from rcbench.measurements.dataset import ReservoirDataset
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

class Measurement():
    """Base class for all measurement types."""
    def __init__(self, path: str):
        """
        Initialize a Measurement object.
        
        Args:
            path (str): Path to the measurement file
        """
        self.path = path
        self.type = self._determine_measurement_type()
        self.dataset: Optional[ReservoirDataset] = None
        self.parser: Optional[MeasurementParser] = None
        self._load_measurement()
    
    def _determine_measurement_type(self) -> MeasurementType:
        """Determine the type of measurement based on the filename."""
        filename = self.path.lower()
        if "nlt" in filename or "nonlinear" in filename:
            return MeasurementType.NLT
        elif "mc" in filename or "memory" in filename:
            return MeasurementType.MEMORY_CAPACITY
        elif "kernel" in filename:
            return MeasurementType.KERNEL_RANK
        elif "activation" in filename:
            return MeasurementType.ACTIVATION
        return MeasurementType.UNKNOWN
    
    def _load_measurement(self):
        """Load the measurement data using MeasurementLoader and MeasurementParser."""
        try:
            from rcbench.measurements.loader import MeasurementLoader
            loader = MeasurementLoader(self.path)
            self.dataset = loader.get_dataset()
            self.parser = MeasurementParser(self.dataset)
            logger.info(f"Successfully loaded measurement of type {self.type.value}")
        except Exception as e:
            logger.error(f"Error loading measurement {self.path}: {str(e)}")
            raise
    
    def get_input_voltages(self):
        """Get input voltages from the measurement."""
        if self.parser:
            return self.parser.get_input_voltages()
        return None
    
    def get_node_voltages(self):
        """Get node voltages from the measurement."""
        if self.parser:
            return self.parser.get_node_voltages()
        return None
    
    def get_time(self):
        """Get time data from the measurement."""
        if self.dataset:
            return self.dataset.time
        return None
    
    def __str__(self):
        return f"Measurement(type={self.type.value}, path={self.path})"

class MemoryCapacity(Measurement):
    """Memory Capacity specific measurement class."""
    def __init__(self, path: str):
        super().__init__(path)
        if self.type != MeasurementType.MEMORY_CAPACITY:
            logger.warning(f"File {path} may not be a memory capacity measurement")
    
