import numpy as np
from scipy.spatial.distance import pdist, squareform
from rcbench.tasks.baseevaluator import BaseEvaluator
from rcbench.logger import get_logger

logger = get_logger(__name__)

class KernelRankEvaluator(BaseEvaluator):
    """
    Evaluates the kernel rank (KR) of reservoir states using Algorithm 2 from Dale et al.
    This evaluator computes the kernel (Gram) matrix from the reservoir states and then
    computes the effective rank by counting the eigenvalues above a threshold.

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
            Relative threshold for counting eigenvalues (eigenvalues > threshold*max_eigenvalue are counted).
            Default is 1e-6.
    """
    def __init__(self, nodes_output, kernel='linear', sigma=1.0, threshold=1e-6):
        self.nodes_output = nodes_output
        self.kernel = kernel
        self.sigma = sigma
        self.threshold = threshold

    def compute_kernel_matrix(self):
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

    def compute_kernel_rank(self):
        """
        Computes the effective kernel rank based on the eigenvalues of the kernel matrix.
        
        Returns:
            effective_rank (int): The effective rank (number of eigenvalues above threshold * max_eigenvalue).
            eigenvalues (np.ndarray): The eigenvalues of the kernel matrix (sorted in ascending order).
        """
        K = self.compute_kernel_matrix()
        # Since K is symmetric, use eigh for numerical stability.
        eigenvalues, _ = np.linalg.eigh(K)
        max_eig = np.max(eigenvalues)
        effective_rank = np.sum(eigenvalues > (self.threshold * max_eig))
        return effective_rank, eigenvalues

    def run_evaluation(self):
        """
        Runs the kernel rank evaluation.
        
        Returns:
            dict: A dictionary containing the effective kernel rank, the eigenvalues, and the kernel parameters.
        """
        rank, eigenvalues = self.compute_kernel_rank()
        logger.info(f"Computed Kernel Rank: {rank}")
        return {
            'kernel_rank': rank,
            'eigenvalues': eigenvalues,
            'kernel': self.kernel,
            'sigma': self.sigma,
            'threshold': self.threshold
        }


