import numpy as np

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

from rcbench.tasks.baseevaluator import BaseEvaluator

from rcbench.logger import get_logger

logger = get_logger(__name__)

def generate_narma_target(u, order=10, coefficients = {'alpha':0.4, 'beta':0.4, 'gamma':0.6, 'delta':0.1}):
    """
    Generates the NARMA target series from an input signal u using a NARMA-10 formulation.
    
    The NARMA-10 system is defined by:
        y[t+1] = 0.3 * y[t] + 0.05 * y[t] * (sum_{i=0}^{order-1} y[t-i]) 
                 + 1.5 * u[t-order] * u[t] + 0.1
                 
    Parameters:
        u (array-like): Input signal.
        order (int): Order of the NARMA system (default is 10).
    
    Returns:
        np.ndarray: The generated NARMA target series.
    """
    N = len(u)
    y = np.zeros(N)

    u = normalize_to_range(u, 0, 0.5)

    alpha = coefficients['alpha']
    beta = coefficients['beta']
    gamma = coefficients['gamma']
    delta = coefficients['delta']

    if order < 2:
        raise ValueError("Unsupported NARMA order. Choose a NARMA order greater than or equal to 2.")
    elif order == 2:
        y[:order-1]=0
        for t in range(2, N):
            y[t] = (alpha * y[t-1] + 
                    beta * y[t-1] * y[t-2] + 
                    gamma * (u[t-1])**3 + 
                    delta)
    # Initialize the first 'order' elements; here they remain zero.
    else:
        y[:order-1]=u[:order-1]
        for t in range(order, N - 1):
            
            y[t + 1] = (alpha * y[t] +
                        beta * y[t] * np.sum(y[t - order:t]) +
                        gamma * u[t - order] * u[t] +
                        delta)
    return y

def normalize_to_range(u, new_min=0.0, new_max=0.5):
    u = np.asarray(u)
    u_min = np.min(u)
    u_max = np.max(u)
    # Avoid division by zero if u_max == u_min:
    if u_max == u_min:
        return np.full(u.shape, new_min)
    return (u - u_min) / (u_max - u_min) * (new_max - new_min) + new_min

class NarmaEvaluator(BaseEvaluator):
    def __init__(self, input_signal, nodes_output, order=2, alpha=0.4, beta=0.4, gamma=0.6, delta=0.1):
        """
        Initializes the NARMA evaluator.

        Parameters:
            input_signal (array-like): The driving input for the NARMA system.
            nodes_output (2D array): The output of the reservoir nodes.
            time_array (array-like): Array of time stamps.
            order (int): The order of the NARMA system (default is 10).
            coefficients (dict): coefficients for the NARMA equation.
        """
        self.input_signal = input_signal
        self.nodes_output = nodes_output
        self.order = order
        self.coefficients = {'alpha' : alpha, 'beta' : beta, 'gamma' : gamma, 'delta' : delta}
        self.targets = self.target_generator()

    def target_generator(self):
        """
        Generates the NARMA target based on the provided input signal.
        Returns a dictionary with key 'narma' mapping to the target series.
        """
        target = generate_narma_target(self.input_signal, self.order, self.coefficients)
        return {'narma': target}
    
    def set_coefficients(self, alpha = 0.4, beta = 0.4, gamma = 0.6, delta = 0.1):
        """
        Sets coefficient for the generation of a NARMA target.
        """
        
        self.coefficients['alpha'] = alpha
        self.coefficients['beta'] = beta
        self.coefficients['gamma'] = gamma
        self.coefficients['delta'] = delta
        

    def evaluate_metric(self, y_true, y_pred, metric='NMSE'):
        """
        Evaluates the performance using the specified metric.
        Supported metrics: 'NMSE', 'RNMSE', and 'MSE'.
        """
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
                       num_features='all',
                       regression_alpha=1.0,
                       train_ratio=0.8,
                       plot=False):
        """
        Runs the NARMA evaluation task. It splits the data into training and testing sets,
        performs feature selection, trains a regression model (Ridge), and returns the evaluation results.

        Parameters:
            metric (str): Performance metric to evaluate the prediction.
            feature_selection_method (str): Method to select features ('kbest' or others).
            num_features (int): Number of features (nodes) to use. If None, all nodes are used.
            regression_alpha (float): Regularization parameter for Ridge regression.
            train_ratio (float): Ratio of data to use for training.
            plot (bool): If True, a prediction plot will be generated.

        Returns:
            dict: A dictionary containing evaluation accuracy, selected features, model, predictions, and true outputs.
        """
        target_waveform = self.targets['narma']
        X = self.nodes_output
        y = target_waveform

        # Train/test split using BaseEvaluator method
        X_train, X_test, y_train, y_test = self.split_train_test(X, y, train_ratio)

        # Feature selection using BaseEvaluator method
        X_train_sel, selected_features = self.feature_selection(
            X_train, y_train, feature_selection_method, num_features
        )
        if feature_selection_method == 'kbest':
            X_test_sel = X_test[:, selected_features]
        else:
            pca = PCA(n_components=num_features)
            pca.fit(X_train)
            X_test_sel = pca.transform(X_test)

        # Train regression model (Ridge)
        model = self.train_regression(X_train_sel, y_train, regression_alpha)
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
