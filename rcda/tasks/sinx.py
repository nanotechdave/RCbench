import numpy as np
from rcda.tasks.baseevaluator import BaseEvaluator
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from rcda.logger import get_logger

logger = get_logger(__name__)
class SinxEvaluator(BaseEvaluator):
    def __init__(self, input_signal, nodes_output):
        """
        Initializes the SinxEvaluator for approximating sin(normalized_input).

        Parameters:
        - input_signal (np.ndarray): Random white noise input signal.
        - nodes_output (np.ndarray): Reservoir node voltages (features).
        """
        super().__init__(input_signal, nodes_output)
        self.normalized_input = self._normalize_input(self.input_signal)
        self.target = np.sin(self.normalized_input)

    def _normalize_input(self, x):
        """Normalize input to range [0, 2π]."""
        x_min = np.min(x)
        x_max = np.max(x)
        return 2 * np.pi * (x - x_min) / (x_max - x_min)

    def evaluate_metric(self, y_true, y_pred, metric='NMSE'):
        if metric == 'NMSE':
            return np.mean((y_true - y_pred) ** 2) / np.var(y_true)
        elif metric == 'RNMSE':
            return np.sqrt(np.mean((y_true - y_pred) ** 2) / np.var(y_true))
        elif metric == 'MSE':
            return mean_squared_error(y_true, y_pred)
        else:
            raise ValueError("Unsupported metric: choose 'NMSE', 'RNMSE', or 'MSE'")

    def run_evaluation(self,
                       metric='NMSE',
                       feature_selection_method='kbest',
                       num_features=10,
                       regression_alpha=1.0,
                       train_ratio=0.8):
        """
        Run the sin(x) reconstruction task using reservoir outputs.

        Returns:
        - dict: result dictionary including accuracy, predictions, model, etc.
        """
        X = self.nodes_output
        y = self.target

        # Train/test split
        X_train, X_test, y_train, y_test = self.split_train_test(X, y, train_ratio)

        # Feature selection
        X_train_sel, selected_features = self.feature_selection(X_train, y_train, feature_selection_method, num_features)
        if feature_selection_method == 'kbest':
            X_test_sel = X_test[:, selected_features]
        else:
            # PCA is fitted inside feature_selection already
            pca = PCA(n_components=num_features)
            pca.fit(X_train)
            X_test_sel = pca.transform(X_test)

        # Regression
        model = self.train_regression(X_train_sel, y_train, alpha=regression_alpha)
        y_pred = model.predict(X_test_sel)

        accuracy = self.evaluate_metric(y_test, y_pred, metric)

        return {
            'accuracy': accuracy,
            'metric': metric,
            'selected_features': selected_features,
            'model': model,
            'y_pred': y_pred,
            'y_test': y_test
        }
