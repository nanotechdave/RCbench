import cProfile
from rcda.measurements.loader import MeasurementLoader
from rcda.measurements.parser import MeasurementParser
from rcda.tasks.memorycapacity import MemoryCapacityEvaluator
from pathlib import Path
# Update the path to your measurement file
BASE_DIR = Path(__file__).resolve().parent
filename = "074_INRiMARC_NWN_Pad129M_gridSE_MemoryCapacity_2024_04_02.txt"

measurement_file = BASE_DIR.parent / "tests" / "test_files" / filename
# Load and parse your measurement data
dataset = MeasurementLoader(measurement_file).get_dataset()
parser = MeasurementParser(dataset)

# Extract necessary signals
input_signal = parser.get_input_voltages()[parser.input_electrodes[0]]
nodes_output = parser.get_node_voltages()

# Initialize the Memory Capacity evaluator
evaluator = MemoryCapacityEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    max_delay=30  # or whatever delay is suitable
)

def run_memory_capacity():
    evaluator.calculate_total_memory_capacity(
        feature_selection_method='kbest',  # or 'pca'
        num_features=None,                 # Use all nodes
        regression_alpha=1.0,
        train_ratio=0.8
    )

# Run the profiler
cProfile.run('run_memory_capacity()', sort='cumtime')
