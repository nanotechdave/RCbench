from pathlib import Path
from typing import List, Dict
from rcbench.classes.Measurement import Measurement, MeasurementType
from rcbench.logger import get_logger

logger = get_logger(__name__)

class Sample():
    """ Sample data class. Instance is an object containing all the measurements of a sample. """
    def __init__(self, name: str, path: str):
        """
        Initialize a Sample object.
        
        Args:
            name (str): Name of the sample
            path (str): Path to the folder containing measurement files
        """
        self.name = name
        self.path = Path(path)
        self.measurements: Dict[str, Measurement] = {}
        self._scan_measurements()
        
    def _scan_measurements(self):
        """Scan the folder for measurement files and initialize measurement objects."""
        if not self.path.exists():
            raise FileNotFoundError(f"Folder not found: {self.path}")
            
        # Find all .txt files that don't contain 'log' in their name
        measurement_files = [f for f in self.path.glob("*.txt") if "log" not in f.name.lower()]
        
        for file_path in measurement_files:
            try:
                # Create appropriate measurement object based on file type
                measurement = Measurement(str(file_path))
                
                # Store the measurement object with the filename as key
                self.measurements[file_path.name] = measurement
                logger.info(f"Loaded measurement: {file_path.name} (type: {measurement.type.value})")
                
            except Exception as e:
                logger.error(f"Error loading measurement {file_path.name}: {str(e)}")
                
    def get_measurement(self, filename: str) -> Measurement:
        """
        Get a specific measurement by filename.
        
        Args:
            filename (str): Name of the measurement file
            
        Returns:
            Measurement: The measurement object
        """
        if filename not in self.measurements:
            raise KeyError(f"Measurement {filename} not found")
        return self.measurements[filename]
    
    def get_measurements_by_type(self, measurement_type: MeasurementType) -> List[Measurement]:
        """
        Get all measurements of a specific type.
        
        Args:
            measurement_type (MeasurementType): Type of measurements to retrieve
            
        Returns:
            List[Measurement]: List of measurements of the specified type
        """
        return [m for m in self.measurements.values() if m.type == measurement_type]
    
    def list_measurements(self) -> List[str]:
        """
        Get a list of all available measurement filenames.
        
        Returns:
            List[str]: List of measurement filenames
        """
        return list(self.measurements.keys())
    
    def __str__(self):
        measurement_types = {}
        for m in self.measurements.values():
            measurement_types[m.type.value] = measurement_types.get(m.type.value, 0) + 1
        
        type_str = ", ".join(f"{t}: {c}" for t, c in measurement_types.items())
        objstr = f"Sample NWN_Pad{self.name} with {len(self.measurements)} measurements ({type_str})"
        return objstr