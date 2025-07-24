__version__ = "0.1.10.dev1"

from .utils import utils
from .tasks.nlt import NltEvaluator
from .tasks.memorycapacity import MemoryCapacityEvaluator
from .tasks.sinx import SinxEvaluator
from .tasks.kernelrank import KernelRankEvaluator
from .tasks.generalizationrank import GeneralizationRankEvaluator
from .tasks.narma import NarmaEvaluator
from .measurements.parser import MeasurementParser
from .measurements.loader import MeasurementLoader
from .measurements.dataset import ReservoirDataset, ElecResDataset
from .logger import get_logger
