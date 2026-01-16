__version__ = "0.1.50"

from .utils import utils
from .tasks.nlt import NltEvaluator
from .tasks.memorycapacity import MemoryCapacityEvaluator
from .tasks.sinx import SinxEvaluator
from .tasks.kernelrank import KernelRankEvaluator
from .tasks.generalizationrank import GeneralizationRankEvaluator
from .tasks.narma import NarmaEvaluator
from .tasks.nonlinearmemory import NonlinearMemoryEvaluator
from .tasks.ipc import IPCEvaluator
from .measurements.parser import MeasurementParser
from .measurements.loader import MeasurementLoader
from .measurements.dataset import ReservoirDataset, ElecResDataset
from .logger import get_logger

# Plot configurations
from .visualization.plot_config import (
    BasePlotConfig,
    NLTPlotConfig,
    MCPlotConfig,
    SinxPlotConfig,
    NarmaPlotConfig,
    NonlinearMemoryPlotConfig,
    IPCPlotConfig,
)
