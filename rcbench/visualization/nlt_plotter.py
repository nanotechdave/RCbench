import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple, Any
import os.path as osp
from rcbench.logger import get_logger
from rcbench.visualization.plot_config import NLTPlotConfig
from rcbench.visualization.base_plotter import BasePlotter

logger = get_logger(__name__)

class NLTPlotter(BasePlotter):
    """
    Plotter for Non-Linear Transformation (NLT) task.
    """

    def __init__(self, config: Optional[NLTPlotConfig] = None):
        """
        Initialize the NLT plotter with configuration.
        
        Args:
            config (Optional[NLTPlotConfig]): Configuration for plotting
        """
        super().__init__(config or NLTPlotConfig())
        self.config = self.config if isinstance(self.config, NLTPlotConfig) else NLTPlotConfig()

    def plot_results(self, 
                    time: np.ndarray,
                    input_signal: np.ndarray,
                    node_outputs: Dict[str, np.ndarray],
                    y_true: Optional[np.ndarray] = None,
                    y_pred: Optional[np.ndarray] = None,
                    target_name: str = "target",
                    save_dir: Optional[str] = None) -> None:
        """
        Generate all relevant plots for the NLT task.
        
        Args:
            time (np.ndarray): Time array 
            input_signal (np.ndarray): Input signal values
            node_outputs (Dict[str, np.ndarray]): Dictionary mapping node names to output arrays
            y_true (Optional[np.ndarray]): True target values
            y_pred (Optional[np.ndarray]): Predicted target values
            target_name (str): Name of the target
            save_dir (Optional[str]): Directory to save plots
        """
        # Always generate general reservoir property plots
        self._plot_general_plots(time, input_signal, node_outputs, save_dir)
            
        # Create general frequency analysis plots if enabled
        if self.config.plot_frequency_analysis:
            self._plot_node_frequency_analysis(time, input_signal, node_outputs, save_dir)
        
        # If target data is provided, also create target-specific plots
        if y_true is not None and y_pred is not None:
            # Create prediction plots if enabled
            if self.config.plot_target_prediction:
                self._plot_prediction_plots(time, y_true, y_pred, target_name, save_dir)
            
            # Create target-specific frequency analysis (comparing target vs prediction)
            if self.config.plot_frequency_analysis:
                # Add target and prediction to signals for frequency analysis
                pred_signals = {
                    "input": input_signal,
                    f"{target_name}_true": y_true,
                    f"{target_name}_pred": y_pred
                }
                
                # Compute frequency analysis
                pred_freqs, pred_spectra = self.compute_frequency_analysis(time, pred_signals)
                
                # Plot frequency comparison
                save_path = None if save_dir is None else osp.join(save_dir, f"{target_name}_frequency.png")
                self.plot_frequency_analysis(
                    pred_freqs, 
                    pred_spectra, 
                    f"Frequency Analysis: {target_name}",
                    save_path,
                    self.config.frequency_range
                )
    
    def _plot_general_plots(self, 
                          time: np.ndarray,
                          input_signal: np.ndarray,
                          node_outputs: Dict[str, np.ndarray],
                          save_dir: Optional[str] = None) -> None:
        """
        Plot general plots including input signal, output responses, and nonlinearity.
        
        Args:
            time (np.ndarray): Time array 
            input_signal (np.ndarray): Input signal values
            node_outputs (Dict[str, np.ndarray]): Dictionary mapping node names to output arrays
            save_dir (Optional[str]): Directory to save plots
        """
        # Plot input signal if enabled
        if self.config.plot_input_signal:
            save_path = None if save_dir is None else osp.join(save_dir, "input_signal.png")
            self.plot_input_signal(time, input_signal, "NLT Task Input Signal", save_path)
            
        # Plot output responses if enabled
        if self.config.plot_output_responses:
            save_path = None if save_dir is None else osp.join(save_dir, "output_responses.png")
            self.plot_output_responses(time, node_outputs, "Reservoir Node Responses", save_path)
            
        # Plot nonlinearity if enabled
        if self.config.plot_nonlinearity:
            save_path = None if save_dir is None else osp.join(save_dir, "nonlinearity.png")
            self.plot_nonlinearity(input_signal, node_outputs, "Node Nonlinearity", save_path, 
                                  style=self.config.nonlinearity_plot_style)
    
    def _plot_prediction_plots(self,
                             time: np.ndarray,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             target_name: str,
                             save_dir: Optional[str] = None) -> None:
        """
        Plot prediction-related plots for the NLT task.
        
        Args:
            time (np.ndarray): Time array
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted target values
            target_name (str): Name of the target
            save_dir (Optional[str]): Directory to save plots
        """
        # Plot target vs prediction
        save_path = None if save_dir is None else osp.join(save_dir, f"{target_name}_prediction.png")
        sample_count = self.config.prediction_sample_count
        title = f"Target vs Prediction: {target_name}"
        self.plot_target_prediction(y_true, y_pred, time, title, save_path, sample_count)
    
    def _plot_node_frequency_analysis(self,
                                   time: np.ndarray,
                                   input_signal: np.ndarray,
                                   node_outputs: Dict[str, np.ndarray],
                                   save_dir: Optional[str] = None) -> None:
        """
        Plot frequency analysis for reservoir nodes (general, not target-specific).
        
        Args:
            time (np.ndarray): Time array
            input_signal (np.ndarray): Input signal values
            node_outputs (Dict[str, np.ndarray]): Dictionary mapping node names to output arrays
            save_dir (Optional[str]): Directory to save plots
        """
        # Compute frequency analysis for input and nodes
        signals = {"input": input_signal}
        # Add a subset of node outputs for analysis
        node_names = list(node_outputs.keys())
        for i, node_name in enumerate(node_names):
            if i >= 5:  # Limit to 5 nodes for clarity
                break
            signals[f"node_{node_name}"] = node_outputs[node_name]
        
        # Run frequency analysis
        frequencies, power_spectra = self.compute_frequency_analysis(time, signals)
        
        # Plot node spectra (input vs nodes)
        save_path = None if save_dir is None else osp.join(save_dir, "node_spectra.png")
        node_spectra = {key: value for key, value in power_spectra.items() if key != "input"}
        self.plot_node_spectra(
            frequencies, 
            power_spectra["input"],
            node_spectra,
            "Node Frequency Response", 
            save_path,
            self.config.frequency_range
        )
    
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
        plt.legend(loc='best')
        plt.grid(True)
        
        self._save_or_show(save_path)
    
    def plot_nonlinearity(self,
                       inputs: np.ndarray,
                       outputs: Dict[str, np.ndarray],
                       title: Optional[str] = None,
                       save_path: Optional[str] = None,
                       style: str = 'scatter',
                       max_plots: int = 5) -> None:
        """
        Plot input-output relationships to visualize nonlinearity.
        
        Args:
            inputs (np.ndarray): Input signal values
            outputs (Dict[str, np.ndarray]): Dictionary mapping node names to output arrays
            title (Optional[str]): Plot title
            save_path (Optional[str]): Path to save the plot
            style (str): Plot style ('scatter' or 'line')
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
            if style == 'scatter':
                plt.scatter(inputs, outputs[node_name], label=node_name, alpha=0.5)
            else:  # 'line' style
                # Sort to ensure proper line plot
                sorted_idx = np.argsort(inputs)
                plt.plot(inputs[sorted_idx], outputs[node_name][sorted_idx], label=node_name)
        
        plt.xlabel('Input Amplitude')
        plt.ylabel('Output Response')
        plt.title(title or 'Input-Output Nonlinearity')
        plt.legend(loc='best')
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
        plt.legend(loc='best')
        plt.grid(True)
        
        self._save_or_show(save_path)
    
    def plot_input_output_spectrum(self,
                                frequencies: np.ndarray,
                                input_spectrum: np.ndarray,
                                output_spectrum: np.ndarray,
                                title: Optional[str] = None,
                                target_name: str = "target",
                                save_path: Optional[str] = None,
                                frequency_range: Optional[Tuple[float, float]] = None) -> None:
        """
        Plot frequency spectra comparison between input and output signals.
        Wrapper around plot_frequency_analysis for backward compatibility.
        
        Args:
            frequencies (np.ndarray): Frequency values
            input_spectrum (np.ndarray): Input signal spectrum
            output_spectrum (np.ndarray): Output/prediction signal spectrum
            title (Optional[str]): Plot title
            target_name (str): Name of the target for labeling
            save_path (Optional[str]): Path to save the plot
            frequency_range (Optional[Tuple[float, float]]): Frequency range to display (min, max)
        """
        # Create dictionary of spectra for the base method
        spectra = {
            "Input": input_spectrum,
            f"Output ({target_name})": output_spectrum
        }
        
        # Forward to base method
        self.plot_frequency_analysis(
            frequencies=frequencies,
            power_spectra=spectra,
            title=title or f"Frequency Analysis: {target_name}",
            save_path=save_path,
            frequency_range=frequency_range
        )
    
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
        Wrapper around base method for backward compatibility.
        
        Args:
            frequencies (np.ndarray): Frequency values
            input_spectrum (np.ndarray): Input signal spectrum
            node_spectra (Dict[str, np.ndarray]): Dictionary mapping node names to spectra
            title (Optional[str]): Plot title
            save_path (Optional[str]): Path to save the plot
            frequency_range (Optional[Tuple[float, float]]): Frequency range to display (min, max)
            max_nodes (int): Maximum number of nodes to include
        """
        # Forward to base method
        super().plot_node_spectra(
            frequencies=frequencies,
            input_spectrum=input_spectrum,
            node_spectra=node_spectra,
            title=title,
            save_path=save_path,
            frequency_range=frequency_range,
            max_nodes=max_nodes
        )
