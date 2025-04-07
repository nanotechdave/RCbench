import numpy as np
from scipy.spatial.distance import pdist, squareform
from rcbench.tasks.baseevaluator import BaseEvaluator
from rcbench.logger import get_logger
from typing import Tuple, Dict, Any

logger = get_logger(__name__)

class KernelRankEvaluator(BaseEvaluator):
    """
    Evaluates the kernel rank (KR) of reservoir states using Algorithm 2 from Dale et al.
    This evaluator computes the kernel (Gram) matrix from the reservoir states and then
    computes the effective rank using Singular Value Decomposition (SVD).

    Parameters:
        nodes_output : np.ndarray
            Reservoir states with shape (T, N), where T is the number of timesteps
            and N is the number of nodes.
        kernel : str, optional
            Type of kernel to use. Options:
              - 'linear': Uses the dot-product kernel, K = X X^T.
              - 'rbf': Uses the Gaussian (RBF) kernel, where
                       K[i,j] = exp(-||x_i - x_j||^2 / (2*sigma^2)).
            Default is 'linear'.
        sigma : float, optional
            Parameter for the RBF kernel (ignored if kernel is 'linear'). Default is 1.0.
        threshold : float, optional
            Relative threshold for counting singular values (values > threshold*max_singular_value are counted).
            Default is 1e-6.
    """
    def __init__(self, 
                 nodes_output: np.ndarray, 
                 kernel: str = 'linear', 
                 sigma: float = 1.0, 
                 threshold: float = 1e-6,
                 ) -> None:
        self.nodes_output: np.ndarray = nodes_output
        self.kernel: str = kernel
        self.sigma: float = sigma
        self.threshold: float = threshold

    def compute_kernel_matrix(self) -> np.ndarray:
        """
        Computes the kernel (Gram) matrix from the reservoir states.
        
        Returns:
            np.ndarray: The computed kernel matrix.
        """
        states = self.nodes_output
        if self.kernel == 'linear':
            # Linear kernel: K = X X^T.
            K = np.dot(states, states.T)
        elif self.kernel == 'rbf':
            # RBF kernel: K[i,j] = exp(-||x_i - x_j||^2 / (2*sigma^2)).
            dists = squareform(pdist(states, 'sqeuclidean'))
            K = np.exp(-dists / (2 * self.sigma**2))
        else:
            raise ValueError("Unsupported kernel type. Please use 'linear' or 'rbf'.")
        return K

    def compute_kernel_rank(self) -> Tuple[int, np.ndarray]:
        """
        Computes the effective kernel rank based on the singular values of the kernel matrix.
        
        Returns:
            effective_rank (int): The effective rank (number of singular values above threshold * max_singular_value).
            singular_values (np.ndarray): The singular values of the kernel matrix (sorted in descending order).
        """
        K = self.compute_kernel_matrix()
        
        # Compute the SVD of the kernel matrix
        U, s, Vh = np.linalg.svd(K, full_matrices=False)
        
        # Calculate effective rank based on singular values
        s_max = np.max(s)
        effective_rank = np.sum(s > (self.threshold * s_max))
        
        return effective_rank, s

    def run_evaluation(self) -> Dict[str, Any]:
        """
        Runs the kernel rank evaluation.
        
        Returns:
            dict: A dictionary containing the effective kernel rank, the singular values, and the kernel parameters.
        """
        rank, singular_values = self.compute_kernel_rank()
        logger.info(f"Computed Kernel Rank: {rank}")
        return {
            'kernel_rank': rank,
            'singular_values': singular_values,
            'kernel': self.kernel,
            'sigma': self.sigma,
            'threshold': self.threshold
        }


