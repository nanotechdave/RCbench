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
from rcbench.visualization.nlt_plotter import NLTPlotter
from rcbench.visualization.plot_config import NLTPlotConfig
from rcbench.logger import get_logger
from typing import Dict, List, Union, Any, Tuple, Optional

logger = get_logger(__name__)
class NltEvaluator(BaseEvaluator):
    def __init__(self, 
                 input_signal: Union[np.ndarray, List[float]], 
                 nodes_output: np.ndarray, 
                 time_array: Union[np.ndarray, List[float]], 
                 waveform_type: str = 'sine',
                 node_names: List[str] = None,
                 plot_config: Optional[NLTPlotConfig] = None,
                 ) -> None:
        """
        Initializes the NLT evaluator.
        
        Parameters:
            input_signal: The input signal array.
            nodes_output: The reservoir nodes output.
            time_array: The time values corresponding to signals.
            waveform_type: Type of waveform, 'sine' (default) or 'triangular'.
            node_names: Optional list of node names.
            plot_config: Optional configuration for plotting.
        """
        # Initialize electrode names if not provided
        if node_names is None:
            node_names = [f'node_{i}' for i in range(nodes_output.shape[1])]
        
        # Call the parent class constructor
        super().__init__(input_signal, nodes_output, node_names)
        
        self.time = time_array
        self.waveform_type = waveform_type
        self.targets: Dict[str, np.ndarray] = self.target_generator()
        
        # Create plotter with config
        self.plotter = NLTPlotter(config=plot_config)

    def _estimate_phase_from_maxima(self, 
                                    signal: np.ndarray, 
                                    time: np.ndarray,
                                    ) -> Tuple[np.ndarray, float]:
        """
        Estimate phase based on time between local maxima.
        Returns a continuous phase vector aligned with the input waveform.
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

        # Create a continuous phase by interpolating between peak positions
        # At each peak, the phase should be 2π*n (where n is the peak number)
        phase_at_peaks = np.arange(len(peaks)) * 2 * np.pi
        
        # Interpolate phase for all time points
        phase = np.interp(time, peak_times, phase_at_peaks)
        
        # Extend the phase linearly beyond the last peak
        if len(time) > 0 and len(peak_times) > 0:
            # Calculate the slope from the last two peaks (or use average frequency)
            if len(peak_times) >= 2:
                slope = 2 * np.pi / avg_period
            else:
                slope = 2 * np.pi * freq
            
            # Extend beyond the last peak
            last_peak_time = peak_times[-1]
            last_peak_phase = phase_at_peaks[-1]
            
            # Update phase for times after the last peak
            beyond_mask = time > last_peak_time
            phase[beyond_mask] = last_peak_phase + slope * (time[beyond_mask] - last_peak_time)
            
            # Update phase for times before the first peak
            before_mask = time < peak_times[0]
            first_peak_time = peak_times[0]
            first_peak_phase = phase_at_peaks[0]
            phase[before_mask] = first_peak_phase + slope * (time[before_mask] - first_peak_time)

        return phase, freq

    def target_generator(self, preserve_scale: bool = True) -> Dict[str, np.ndarray]:
        """
        Generate nonlinear targets using a more robust approach.
        Instead of complex phase estimation, use the Hilbert transform for better results.
        """
        from scipy.signal import hilbert
        
        # Normalize input signal for target generation
        signal = self.input_signal - np.mean(self.input_signal)
        signal /= np.max(np.abs(signal))

        targets = {}

        # 1. Square wave: sign of normalized input
        targets['square_wave'] = np.sign(signal)

        # 2. Pi/2 shifted sine: Use Hilbert transform to get quadrature component
        # This is much more robust than phase estimation
        analytic_signal = hilbert(signal)
        targets['pi_half_shifted'] = np.imag(analytic_signal)  # Quadrature component

        # 3. Double frequency: Use the square of the signal minus DC component
        # For sin(ωt), sin²(ωt) = (1 - cos(2ωt))/2, so we get double frequency
        signal_squared = signal**2
        targets['double_frequency'] = signal_squared - np.mean(signal_squared)

        # 4. Triangle from sine: Use proper triangular wave generation
        if self.waveform_type == 'sine':
            # Method 1: Use the Hilbert transform to get instantaneous phase
            analytic_signal = hilbert(signal)
            instantaneous_phase = np.angle(analytic_signal)
            # Unwrap phase to make it continuous
            instantaneous_phase = np.unwrap(instantaneous_phase)
            # Generate triangular wave using sawtooth with width=0.5 (symmetric triangle)
            targets['triangular_wave'] = sawtooth(instantaneous_phase, width=0.5)

        # 5. Sine from triangle
        if self.waveform_type == 'triangular':
            # If input is triangular, approximate sine using the fundamental frequency
            # Use Fourier approach or simple mapping
            targets['sine_wave'] = signal  # For now, keep as is (can be improved)

        # Scale preservation
        if preserve_scale:
            input_min = np.min(self.input_signal)
            input_max = np.max(self.input_signal)

            for key in targets:
                target = targets[key]
                # Normalize to [0, 1]
                target_min = np.min(target)
                target_max = np.max(target)
                if target_max > target_min:  # Avoid division by zero
                    norm = (target - target_min) / (target_max - target_min)
                    # Rescale to match input range
                    targets[key] = norm * (input_max - input_min) + input_min
                else:
                    targets[key] = np.full_like(target, np.mean([input_min, input_max]))

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
                       modeltype: str = "Ridge",
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
        
        model = self.train_regression(X_train_sel, y_train, modeltype, regression_alpha)
        y_pred = model.predict(X_test_sel)
        accuracy = self.evaluate_metric(y_test, y_pred, metric)
        
        # Update plot config with the train_ratio used
        self.plotter.config.train_ratio = train_ratio

        if plot:
            # Create a dictionary of node outputs
            node_outputs = {}
            for i, name in enumerate(self.electrode_names):
                node_outputs[name] = self.nodes_output[:, i]
            
            # Generate frequency analysis data if needed
            frequencies = None
            power_spectra = None
            
            # Plot using our NLTPlotter
            self.plotter.plot_input_signal(
                time=self.time,
                input_signal=self.input_signal,
                title=f"Input Signal for {target_name}",
                save_path=self.plotter.config.get_save_path(f"{target_name}_input.png")
            )
            
            # Plot prediction results
            # Calculate proper test indices based on actual test data length
            test_start = int(train_ratio * len(y))
            test_length = len(y_test)
            
            # Ensure we don't go beyond available data
            if test_start + test_length > len(self.time):
                test_start = len(self.time) - test_length
            
            test_indices = np.arange(test_start, test_start + test_length)
            test_time = self.time[test_indices]
            
            self.plotter.plot_target_prediction(
                y_true=y_test,
                y_pred=y_pred,
                time=test_time,
                title=f"NLT Task: {target_name}",
                save_path=self.plotter.config.get_save_path(f"{target_name}_prediction.png")
            )
        
        return {
            'accuracy': accuracy,
            'metric': metric,
            'selected_features': selected_features,
            'model': model,
            'y_pred': y_pred,
            'y_test': y_test,
            'y_train': y_train,
            'train_ratio': train_ratio,
        }
        
    def plot_results(self, existing_results: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        Generate plots for all targets based on plotter's configuration.
        
        Args:
            existing_results: Optional dictionary of pre-computed results for each target
        """
        if not self.targets:
            logger.warning("No targets available for plotting.")
            return
            
        # Use existing results if provided, otherwise compute them
        results = {}
        if existing_results:
            results = existing_results
        else:
            # We need to run evaluation for each target first to get predictions
            # Use train_ratio from config
            for target_name in self.targets:
                try:
                    result = self.run_evaluation(
                        target_name=target_name,
                        feature_selection_method='pca',
                        num_features='all',
                        train_ratio=self.plotter.config.train_ratio,
                        plot=False  # Don't plot individually
                    )
                    results[target_name] = result
                except Exception as e:
                    logger.error(f"Error evaluating {target_name}: {str(e)}")
        
        # Create node outputs dictionary for visualization
        node_outputs = {}
        for i, name in enumerate(self.node_names):
            node_outputs[name] = self.nodes_output[:, i]
        
        # First create general plots with the full dataset (not target-specific)
        self.plotter.plot_results(
            time=self.time,
            input_signal=self.input_signal,
            node_outputs=node_outputs,
            save_dir=self.plotter.config.save_dir
        )
        
        # Generate plots for each target with the appropriate test data
        for target_name, result in results.items():
            if 'y_pred' in result and 'y_test' in result:
                # For each target, get correct time array for test data
                
                # Get train_ratio: first from result, then from config
                if 'train_ratio' in result:
                    train_ratio = result['train_ratio']
                elif 'y_train' in result and len(result.get('y_train', [])) > 0:
                    train_ratio = len(result['y_train']) / (len(result['y_train']) + len(result['y_test']))
                else:
                    train_ratio = self.plotter.config.train_ratio
                
                # Calculate test indices based on the actual test data length
                total_samples = len(self.time)
                test_start = int(train_ratio * total_samples)
                test_length = len(result['y_test'])
                
                # Ensure we don't go beyond the available data
                if test_start + test_length > total_samples:
                    test_start = total_samples - test_length
                
                # Create consistent test indices
                test_indices = np.arange(test_start, test_start + test_length)
                
                # Get time values for test data - use actual time values, not artificial ones
                test_time = self.time[test_indices]
                
                # Set save directory for this target's plots
                save_dir = self.plotter.config.save_dir
                
                # Use test data slices that are consistent with test_indices
                test_input_signal = self.input_signal[test_indices]
                test_node_outputs = {}
                for node_name, output in node_outputs.items():
                    test_node_outputs[node_name] = output[test_indices]
                
                # Use the unified plotting method for each target
                self.plotter.plot_results(
                    time=test_time,
                    input_signal=test_input_signal,
                    node_outputs=test_node_outputs,
                    y_true=result['y_test'],
                    y_pred=result['y_pred'],
                    target_name=target_name,
                    save_dir=save_dir
                )
