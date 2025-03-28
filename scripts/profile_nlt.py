import cProfile
from rcda.measurements.loader import MeasurementLoader
from rcda.measurements.parser import MeasurementParser
from rcda.tasks.nlt import NltEvaluator
from pathlib import Path
# Load and parse your real measurement data
BASE_DIR = Path(__file__).resolve().parent
filename = "837_INRiMJanis_NWN_Pad99C_grid_SE_Custom Wave Measurement_2024_02_01.txt"

measurement_file = BASE_DIR.parent / "tests" / "test_files" / filename
dataset = MeasurementLoader(measurement_file).get_dataset()
parser = MeasurementParser(dataset)

input_signal = parser.get_input_voltages()[parser.input_electrodes[0]]
nodes_output = parser.get_node_voltages()
time_array = dataset.time

evaluator = NltEvaluator(input_signal, nodes_output, time_array, waveform_type='sine')

def run_eval():
    evaluator.run_evaluation(
        target_name='square_wave',
        metric='NMSE',
        num_features=None,
        plot=False
    )

cProfile.run('run_eval()', sort='cumtime')
