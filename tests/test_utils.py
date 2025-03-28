from rcbench.utils.utils import train_test_split_time_series
import numpy as np

def test_train_test_split_time_series() -> None:
    data = np.ones((100,16))
    target = np.ones(100)
    data_train, data_test, target_train, target_test = train_test_split_time_series(data, target, 0.3)
    isDataCorrect = data_train.shape == (70,16) and data_test.shape == (30,16)
    isTargetCorrect = target_train.shape == (70,) and target_test.shape == (30,)
    assert isDataCorrect and isTargetCorrect