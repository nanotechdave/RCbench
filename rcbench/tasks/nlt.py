import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from scipy.signal import sawtooth
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from rcbench.tasks.baseevaluator import BaseEvaluator
from rcbench.tasks.featureselector import FeatureSelector
from rcbench.visualization.nlt_plotter import plot_nlt_prediction
from rcbench.logger import get_logger
from typing import Dict, List, Union, Any, Tuple

logger = get_logger(__name__)
class NltEvaluator(BaseEvaluator):
    def __init__(self, 
                 input_signal: Union[np.ndarray, List[float]], 
                 nodes_output: np.ndarray, 
                 time_array: Union[np.ndarray, List[float]], 
                 waveform_type: str = 'sine',
                 electrode_names: List[str] = None,
                 ) -> None:
        """
        Initializes the NLT evaluator.
        
        Parameters:
            input_signal: The input signal array.
            nodes_output: The reservoir nodes output.
            time_array: The time values corresponding to signals.
            waveform_type: Type of waveform, 'sine' (default) or 'triangular'.
            electrode_names: Optional list of electrode names.
        """
        # Initialize electrode names if not provided
        if electrode_names is None:
            electrode_names = [f'el_{i}' for i in range(nodes_output.shape[1])]
        
        # Call the parent class constructor
        super().__init__(input_signal, nodes_output, electrode_names)
        
        self.time = time_array
        self.waveform_type = waveform_type
        self.targets: Dict[str, np.ndarray] = self.target_generator()

        
    def _estimate_phase_from_maxima(self, 
                                    signal: np.ndarray, 
                                    time: np.ndarray,
                                    ) -> Tuple[np.ndarray, float]:
        """
        Estimate phase based on time between local maxima.
        Returns a phase vector aligned with the input waveform.
        """
        # Normalize and center the signal
        signal = signal - np.mean(signal)
        signal = signal / np.max(np.abs(signal))

        # Detect peaks
        peaks, _ = find_peaks(signal)
        peak_times = time[peaks]

        if len(peak_times) < 2:
            raise ValueError("Not enough peaks found to estimate frequency.")

        # Estimate period as average time between peaks
        periods = np.diff(peak_times)
        avg_period = np.mean(periods)
        freq = 1 / avg_period

        # Now build a linear phase ramp between peaks
        phase = np.zeros_like(signal)
        for i in range(len(peaks) - 1):
            start = peaks[i]
            end = peaks[i + 1]
            phase[start:end] = np.linspace(0, 2 * np.pi, end - start, endpoint=False)

        # Handle tail
        if peaks[-1] < len(signal) - 1:
            tail_len = len(signal) - peaks[-1]
            phase[peaks[-1]:] = np.linspace(0, 2 * np.pi, tail_len, endpoint=False)

        return np.unwrap(phase), freq

    def target_generator(self, preserve_scale: bool = True) -> Dict[str, np.ndarray]:
        """
        Generate nonlinear targets aligned with waveform maxima.
        """
        phase, freq = self._estimate_phase_from_maxima(self.input_signal, self.time)
        signal = self.input_signal - np.mean(self.input_signal)
        signal /= np.max(np.abs(signal))

        targets = {}

        # 1. Square wave: sign of normalized input
        targets['square_wave'] = np.sign(signal)

        # 2. Pi/2 shifted sine
        targets['pi_half_shifted'] = np.sin(phase)

        # 3. Double frequency: sin(2ϕ)
        targets['double_frequency'] = -np.sin((2 * phase))

        # 4. Triangle from sine
        if self.waveform_type == 'sine':
            targets['triangular_wave'] = -sawtooth(phase, width=0.5)

        # 5. Sine from triangle
        if self.waveform_type == 'triangular':
            targets['sine_wave'] = np.sin(phase)

        if preserve_scale:
            input_min = np.min(self.input_signal)
            input_max = np.max(self.input_signal)

            for key in targets:
                target = targets[key]
                # Normalize to [0, 1]
                norm = (target - np.min(target)) / (np.max(target) - np.min(target))
                # Rescale to match input range
                targets[key] = norm * (input_max - input_min) + input_min

        return targets

    def evaluate_metric(self, 
                        y_true: np.ndarray, 
                        y_pred: np.ndarray, 
                        metric: str = 'NMSE',
                        ) -> float:
        if metric == 'NMSE':
            return np.mean((y_true - y_pred) ** 2) / np.var(y_true)
        elif metric == 'RNMSE':
            return np.sqrt(np.mean((y_true - y_pred) ** 2) / np.var(y_true))
        elif metric == 'MSE':
            return mean_squared_error(y_true, y_pred)
        else:
            raise ValueError("Unsupported metric: choose 'NMSE', 'RNMSE', or 'MSE'")

    def run_evaluation(self,
                       target_name: str,
                       metric: str = 'NMSE',
                       feature_selection_method: str = 'kbest',
                       num_features: Union[str, int] = 'all',
                       regression_alpha: float =1.0,
                       train_ratio: float = 0.8,
                       plot: bool = False
                       ) -> Dict[str, Any]:

        if target_name not in self.targets:
            raise ValueError(f"Target '{target_name}' not found. Available: {list(self.targets)}")

        target_waveform = self.targets[target_name]
        X = self.nodes_output
        y = target_waveform
        

        # Train/test split
        X_train, X_test, y_train, y_test = self.split_train_test(X, y, train_ratio)

        # Feature selection
        X_train_sel, selected_features, _ = self.feature_selection(
            X_train, y_train, feature_selection_method, num_features
        )
        if feature_selection_method == 'kbest':
            X_test_sel = X_test[:, selected_features]
        else:
            X_test_sel = self.apply_feature_selection(X_test)
        
        model = self.train_regression(X_train_sel, y_train, regression_alpha)
        y_pred = model.predict(X_test_sel)
        accuracy = self.evaluate_metric(y_test, y_pred, metric)

        if plot:
            plot_nlt_prediction(
                input_signal=self.input_signal,
                target=target_waveform,
                prediction=y_pred,
                time=self.time,
                train_ratio=train_ratio,
                title=f"NLT Task: {target_name}"
            )
        
        return {
            'accuracy': accuracy,
            'metric': metric,
            'selected_features': selected_features,
            'model': model,
            'y_pred': y_pred,
            'y_test': y_test,
        }
