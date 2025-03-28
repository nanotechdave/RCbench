

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from rcbench.tasks.baseevaluator import BaseEvaluator
from rcbench.logger import get_logger
from rcbench.tasks.c_metrics import evaluate_mc

    


logger = get_logger(__name__)
class MemoryCapacityEvaluator(BaseEvaluator):
    def __init__(self, input_signal, nodes_output, max_delay=30):
        """
        Initializes the Memory Capacity evaluator.

        Parameters:
        - input_signal (np.ndarray): Input stimulation signal array.
        - nodes_output (np.ndarray): Reservoir node output (features).
        - max_delay (int): Maximum delay steps to evaluate.
        """
        self.input_signal = input_signal
        self.nodes_output = nodes_output
        self.max_delay = max_delay
        self.targets = self.target_generator()

    def target_generator(self):
        """
        Generates delayed versions of the input signal.

        Returns:
        - dict: delay (int) -> delayed input (np.ndarray)
        """
        targets = {}
        for delay in range(1, self.max_delay + 1):
            targets[delay] = np.roll(self.input_signal, delay)
        return targets

    """ def evaluate_mc(self, y_true, y_pred):
        covariance = np.cov(y_pred, y_true)[0, 1]
        variance_pred = np.var(y_pred)
        variance_true = np.var(y_true)
        if variance_true == 0 and variance_pred == 0:
            return 1.0
        else:
            return covariance**2 / (variance_pred * variance_true) """
    
    def run_evaluation(self,
                       delay,
                       feature_selection_method='kbest',
                       num_features=10,
                       regression_alpha=1.0,
                       train_ratio=0.8):
        if delay not in self.targets:
            raise ValueError(f"Delay '{delay}' not available.")

        target_waveform = self.targets[delay]
        
        # Adjust data for delay
        X = self.nodes_output[delay:]
        y = target_waveform[delay:]

        # Split data
        split_idx = int(train_ratio * len(y))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Feature selection
        X_train_selected, selected_features = self.feature_selection(
            X_train, y_train,
            method=feature_selection_method,
            num_features=num_features
        )

        # Apply selection to test data
        if feature_selection_method == 'kbest':
            X_test_selected = X_test[:, selected_features]
        else:
            pca = PCA(n_components=num_features)
            pca.fit(X_train)
            X_test_selected = pca.transform(X_test)

        # Regression model
        model = Ridge(alpha=regression_alpha)
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)

        # Evaluate MC
        mc = evaluate_mc(y_test, y_pred)

        return {
            'delay': delay,
            'memory_capacity': mc,
            'selected_features': selected_features,
            'model': model,
            'y_pred': y_pred,
            'y_test': y_test,
        }

    def calculate_total_memory_capacity(self,
                                        feature_selection_method='kbest',
                                        num_features=10,
                                        regression_alpha=1.0,
                                        train_ratio=0.8):
        total_mc = 0
        delay_results = {}
        for delay in range(1, self.max_delay + 1):
            result = self.run_evaluation(
                delay=delay,
                feature_selection_method=feature_selection_method,
                num_features=num_features,
                regression_alpha=regression_alpha,
                train_ratio=train_ratio
            )
            total_mc += result['memory_capacity']
            delay_results[delay] = result['memory_capacity']

        return {
            'total_memory_capacity': total_mc,
            'delay_results': delay_results
        }
