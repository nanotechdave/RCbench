import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Union, Tuple

def plot_nlt_prediction(input_signal: np.ndarray, 
                        target: np.ndarray, 
                        prediction: np.ndarray, 
                        time: Optional[Union[np.ndarray, List]] = None, 
                        train_ratio: float = 0.8, 
                        title: str = 'NLT Task', 
                        figsize: Tuple[int, int] = (10, 4),
                        ) -> None:
    """
    Plots input signal, target waveform, and prediction for an NLT task.

    Parameters:
    - input_signal (np.ndarray): Original input signal to the reservoir.
    - target (np.ndarray): Ground truth waveform (e.g. sin(x), square wave, etc.).
    - prediction (np.ndarray): Model prediction on the test set.
    - time (np.ndarray or None): Time axis, or None to use index.
    - train_ratio (float): Where to place the train/test split marker.
    - title (str): Plot title.
    - figsize (tuple): Size of the figure.
    """
    N = len(target)
    split_idx = int(train_ratio * N)
    x = np.arange(N) if time is None else time
    test_x = x[split_idx:]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, target, label='Target', color='black', linewidth=1.5)
    ax.plot(x, input_signal, label='Input Signal', color='tab:blue', alpha=0.3)
    ax.plot(test_x, prediction, label='Prediction', color='tab:red', linestyle='--')
    ax.axvline(x[split_idx], color='gray', linestyle=':', label='Train/Test Split')

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Signal")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
