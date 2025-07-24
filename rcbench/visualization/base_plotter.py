import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any, Tuple
from rcbench.logger import get_logger
from rcbench.visualization.plot_config import BasePlotConfig

logger = get_logger(__name__)

class BasePlotter:
    """Base class for all plotters with common functionality."""
    
    def __init__(self, config: Optional[BasePlotConfig] = None):
        """
        Initialize the BasePlotter.
        
        Args:
            config (Optional[BasePlotConfig]): Configuration for the plotter
        """
        self.config = config or BasePlotConfig()
        
        try:
            # Use one of the available seaborn styles based on newer matplotlib versions
            plt.style.use('default')  # Use default style to avoid legend conflicts
        except OSError:
            logger.warning("Seaborn style not available. Using default matplotlib style.")
            plt.style.use('default')
    
    @property
    def figsize(self):
        return self.config.figsize
    
    @figsize.setter
    def figsize(self, value):
        self.config.figsize = value
    
    def _create_figure(self) -> plt.Figure:
        """Create a figure with the configured size."""
        return plt.figure(figsize=self.figsize)
        
    def _save_or_show(self, save_path: Optional[str] = None) -> None:
        """
        Save the current figure to a file or display it based on configuration.
        
        Args:
            save_path (Optional[str]): Path to save the figure
        """
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
            plt.close()
        elif self.config.show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_target_prediction(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            time: Optional[np.ndarray] = None,
                            title: Optional[str] = None,
                            save_path: Optional[str] = None,
                            sample_count: Optional[int] = None,
                            x_label: str = 'Time',
                            y_label: str = 'Value') -> None:
        """
        Plot true vs predicted values - common across many RC tasks.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            time (Optional[np.ndarray]): Time array for x-axis
            title (Optional[str]): Plot title
            save_path (Optional[str]): Path to save the plot
            sample_count (Optional[int]): Number of samples to show (for large datasets)
            x_label (str): Label for x-axis
            y_label (str): Label for y-axis
        """
        if time is None:
            time = np.arange(len(y_true))
            
        # Sample data if needed to prevent overcrowded plots
        if sample_count and sample_count < len(y_true):
            indices = np.linspace(0, len(y_true)-1, sample_count, dtype=int)
            time = time[indices]
            y_true = y_true[indices]
            y_pred = y_pred[indices]
        
        self._create_figure()
        plt.plot(time, y_true, 'b-', label='True', alpha=0.7)
        plt.plot(time, y_pred, 'r--', label='Predicted', alpha=0.7)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title or 'True vs Predicted Values')
        legend = plt.legend()
        legend.get_frame().set_visible(True)  # Force frame to be visible
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_alpha(1.0)
        legend.get_frame().set_linewidth(1.0)  # Set border width
        plt.grid(True)
        
        self._save_or_show(save_path)
    
    def plot_feature_importance(self,
                              feature_importance: np.ndarray,
                              feature_names: Optional[List[str]] = None,
                              title: Optional[str] = None,
                              save_path: Optional[str] = None) -> None:
        """
        Plot feature importance scores - common across many RC tasks.
        
        Args:
            feature_importance (np.ndarray): Array of feature importance scores
            feature_names (Optional[List[str]]): List of feature names
            title (Optional[str]): Plot title
            save_path (Optional[str]): Path to save the plot
        """
        n_features = len(feature_importance)
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(n_features)]
        
        # Sort features by importance for consistent display
        sorted_indices = np.argsort(feature_importance)[::-1]  # Sort in descending order
        sorted_importance = feature_importance[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        self._create_figure()
        plt.bar(range(n_features), sorted_importance)
        plt.xticks(range(n_features), sorted_names, rotation=45)
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.title(title or 'Feature Importance')
        plt.tight_layout()
        
        self._save_or_show(save_path)

    def plot_heatmap(self,
                    data: np.ndarray,
                    x_labels: Optional[List[Any]] = None,
                    y_labels: Optional[List[Any]] = None,
                    title: Optional[str] = None,
                    save_path: Optional[str] = None,
                    colormap: str = 'viridis',
                    x_label: str = 'X',
                    y_label: str = 'Y',
                    colorbar_label: str = 'Value') -> None:
        """
        Plot data as a heatmap - common across many RC tasks.
        
        Args:
            data (np.ndarray): 2D array of values
            x_labels (Optional[List[Any]]): Labels for x-axis
            y_labels (Optional[List[Any]]): Labels for y-axis
            title (Optional[str]): Plot title
            save_path (Optional[str]): Path to save the plot
            colormap (str): Matplotlib colormap to use
            x_label (str): Label for x-axis
            y_label (str): Label for y-axis
            colorbar_label (str): Label for colorbar
        """
        self._create_figure()
        plt.imshow(data, aspect='auto', cmap=colormap)
        plt.colorbar(label=colorbar_label)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title or 'Heatmap')
        
        # Add axis labels if provided
        if x_labels is not None:
            plt.xticks(range(len(x_labels)), x_labels)
        if y_labels is not None:
            plt.yticks(range(len(y_labels)), y_labels)
        
        self._save_or_show(save_path)
        
    def plot_input_signal(self, 
                        time: np.ndarray,
                        input_signal: np.ndarray,
                        title: Optional[str] = None,
                        save_path: Optional[str] = None) -> None:
        """
        Plot input signal.
        
        Args:
            time (np.ndarray): Time array
            input_signal (np.ndarray): Input signal values
            title (Optional[str]): Plot title
            save_path (Optional[str]): Path to save the plot
        """
        self._create_figure()
        plt.plot(time, input_signal, 'b-', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Input Amplitude')
        plt.title(title or 'Input Signal')
        plt.grid(True)
        
        self._save_or_show(save_path)
    
    def plot_output_responses(self,
                           time: np.ndarray,
                           outputs: Dict[str, np.ndarray],
                           title: Optional[str] = None,
                           save_path: Optional[str] = None,
                           max_plots: int = 10) -> None:
        """
        Plot output responses for various nodes.
        
        Args:
            time (np.ndarray): Time array
            outputs (Dict[str, np.ndarray]): Dictionary mapping node names to output arrays
            title (Optional[str]): Plot title
            save_path (Optional[str]): Path to save the plot
            max_plots (int): Maximum number of outputs to plot
        """
        self._create_figure()
        
        # Limit number of plots for clarity
        if len(outputs) > max_plots:
            logger.warning(f"Too many outputs to plot clearly, limiting to {max_plots}")
            node_names = list(outputs.keys())[:max_plots]
        else:
            node_names = list(outputs.keys())
        
        for node_name in node_names:
            plt.plot(time, outputs[node_name], label=node_name)
        
        plt.xlabel('Time')
        plt.ylabel('Output Response')
        plt.title(title or 'Node Responses')
        legend = plt.legend(loc='best')
        legend.get_frame().set_visible(True)  # Force frame to be visible
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_alpha(1.0)
        legend.get_frame().set_linewidth(1.0)  # Set border width
        plt.grid(True)
        
        self._save_or_show(save_path)
    
    def plot_nonlinearity(self,
                       inputs: np.ndarray,
                       outputs: Dict[str, np.ndarray],
                       title: Optional[str] = None,
                       save_path: Optional[str] = None,
                       style: Optional[str] = None,
                       max_plots: int = 5) -> None:
        """
        Plot input-output relationships to visualize nonlinearity.
        
        Args:
            inputs (np.ndarray): Input signal values
            outputs (Dict[str, np.ndarray]): Dictionary mapping node names to output arrays
            title (Optional[str]): Plot title
            save_path (Optional[str]): Path to save the plot
            style (Optional[str]): Plot style ('scatter' or 'line'), defaults to config value
            max_plots (int): Maximum number of outputs to plot
        """
        self._create_figure()
        
        # Use style from config if not specified
        style = style or self.config.nonlinearity_plot_style
        
        # Limit number of plots for clarity
        if len(outputs) > max_plots:
            logger.warning(f"Too many outputs to plot clearly, limiting to {max_plots}")
            node_names = list(outputs.keys())[:max_plots]
        else:
            node_names = list(outputs.keys())
        
        for node_name in node_names:
            if style == 'scatter':
                plt.scatter(inputs, outputs[node_name], label=node_name, alpha=0.5)
            else:  # 'line' style
                # Sort to ensure proper line plot
                sorted_idx = np.argsort(inputs)
                plt.plot(inputs[sorted_idx], outputs[node_name][sorted_idx], label=node_name)
        
        plt.xlabel('Input Amplitude')
        plt.ylabel('Output Response')
        plt.title(title or 'Input-Output Nonlinearity')
        legend = plt.legend(loc='best')
        legend.get_frame().set_visible(True)  # Force frame to be visible
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_alpha(1.0)
        legend.get_frame().set_linewidth(1.0)  # Set border width
        plt.grid(True)
        
        self._save_or_show(save_path)
    
    def plot_node_spectra(self,
                        frequencies: np.ndarray,
                        input_spectrum: np.ndarray,
                        node_spectra: Dict[str, np.ndarray],
                        title: Optional[str] = None,
                        save_path: Optional[str] = None,
                        frequency_range: Optional[Tuple[float, float]] = None,
                        max_nodes: int = 5) -> None:
        """
        Plot frequency spectra of input vs multiple node outputs.
        
        Args:
            frequencies (np.ndarray): Frequency values
            input_spectrum (np.ndarray): Input signal spectrum
            node_spectra (Dict[str, np.ndarray]): Dictionary mapping node names to spectra
            title (Optional[str]): Plot title
            save_path (Optional[str]): Path to save the plot
            frequency_range (Optional[Tuple[float, float]]): Frequency range to display (min, max)
            max_nodes (int): Maximum number of nodes to include
        """
        self._create_figure()
        
        # Use frequency range from config if not specified
        if frequency_range is None:
            frequency_range = self.config.frequency_range
        
        # Limit frequency range for display
        if frequency_range:
            min_freq, max_freq = frequency_range
            mask = (frequencies >= min_freq) & (frequencies <= max_freq)
            frequencies = frequencies[mask]
            input_spectrum = input_spectrum[mask]
        else:
            mask = slice(None)  # Include all frequencies
        
        # Plot input spectrum
        plt.semilogy(frequencies, input_spectrum, label='Input Signal', color='black', linewidth=2)
        
        # Limit number of node plots for clarity
        if len(node_spectra) > max_nodes:
            logger.warning(f"Too many nodes to plot clearly, limiting to {max_nodes}")
            node_names = list(node_spectra.keys())[:max_nodes]
        else:
            node_names = list(node_spectra.keys())
        
        # Plot node spectra
        for node_name in node_names:
            plt.semilogy(frequencies, node_spectra[node_name][mask], label=node_name, alpha=0.7)
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectrum (log scale)')
        plt.title(title or 'Node Frequency Response Analysis')
        legend = plt.legend(loc='best')
        legend.get_frame().set_visible(True)  # Force frame to be visible
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_alpha(1.0)
        legend.get_frame().set_linewidth(1.0)  # Set border width
        plt.grid(True)
        
        self._save_or_show(save_path)
    
    def plot_frequency_analysis(self,
                             frequencies: np.ndarray,
                             power_spectra: Dict[str, np.ndarray],
                             title: Optional[str] = None,
                             save_path: Optional[str] = None,
                             frequency_range: Optional[Tuple[float, float]] = None,
                             max_plots: int = 5) -> None:
        """
        Plot frequency spectra for multiple signals.
        
        Args:
            frequencies (np.ndarray): Frequency values
            power_spectra (Dict[str, np.ndarray]): Dictionary mapping signal names to power spectra
            title (Optional[str]): Plot title
            save_path (Optional[str]): Path to save the plot
            frequency_range (Optional[Tuple[float, float]]): Frequency range to display (min, max)
            max_plots (int): Maximum number of spectra to plot
        """
        self._create_figure()
        
        # Use frequency range from config if not specified
        if frequency_range is None:
            frequency_range = self.config.frequency_range
        
        # Limit frequency range for display
        if frequency_range:
            min_freq, max_freq = frequency_range
            mask = (frequencies >= min_freq) & (frequencies <= max_freq)
            frequencies = frequencies[mask]
        else:
            mask = slice(None)  # Include all frequencies
        
        # Limit number of plots for clarity
        if len(power_spectra) > max_plots:
            logger.warning(f"Too many spectra to plot clearly, limiting to {max_plots}")
            signal_names = list(power_spectra.keys())[:max_plots]
        else:
            signal_names = list(power_spectra.keys())
        
        for signal_name in signal_names:
            plt.semilogy(frequencies, power_spectra[signal_name][mask], label=signal_name)
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectrum')
        plt.title(title or 'Frequency Analysis')
        legend = plt.legend(loc='best')
        legend.get_frame().set_visible(True)  # Force frame to be visible
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_alpha(1.0)
        legend.get_frame().set_linewidth(1.0)  # Set border width
        plt.grid(True)
        
        self._save_or_show(save_path)
    
    def compute_frequency_analysis(self,
                               time: np.ndarray,
                               signals: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute frequency analysis for multiple signals.
        
        Args:
            time (np.ndarray): Time array 
            signals (Dict[str, np.ndarray]): Dictionary mapping signal names to time-domain signals
            
        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray]]: 
                - Frequency axis (Hz)
                - Dictionary mapping signal names to power spectra
        """
        # Compute sampling rate and time parameters
        sample_rate = 1.0 / (time[1] - time[0]) if len(time) > 1 else 1.0
        n_samples = len(time)
        
        # Calculate frequency axis (positive frequencies only)
        freqs = np.fft.fftfreq(n_samples, 1/sample_rate)
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        
        # Compute FFT and power spectrum for each signal
        power_spectra = {}
        for signal_name, signal in signals.items():
            # Check that signal matches expected length
            if len(signal) != n_samples:
                logger.warning(f"Signal '{signal_name}' length {len(signal)} doesn't match time array length {n_samples}")
                continue
                
            # Calculate FFT
            fft = np.fft.fft(signal)
            # Calculate power spectral density
            psd = np.abs(fft)**2 / n_samples
            # Keep only positive frequencies
            power_spectra[signal_name] = psd[pos_mask]
            
        return freqs, power_spectra 