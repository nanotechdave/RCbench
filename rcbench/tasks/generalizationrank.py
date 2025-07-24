import numpy as np
from rcbench.tasks.baseevaluator import BaseEvaluator
from rcbench.logger import get_logger
from typing import Dict, Tuple, Any

logger = get_logger(__name__)

class GeneralizationRankEvaluator(BaseEvaluator):
    """
    Evaluates the Generalization Rank (GR) of a reservoir from multiple noisy input streams,
    following the procedure described in I T Vidamour et al 2022 Nanotechnology 33 485203

    Procedure:
      1. Supply m distinct input streams u₁, …, uₘ to the reservoir. Each produces an
         n-dimensional reservoir state x_ui.
      2. Construct the matrix M by placing each state x_ui as a column. (Thus, M has shape (n, m).)
          Note: If the input is provided as an (m, n) array (each row is a state vector),
          it will be transposed.
      3. Compute the singular value decomposition (SVD) of M.
      4. The effective (generalization) rank is given by the number of singular values sᵢ
         satisfying
                sᵢ > threshold × s_max,
         where s_max is the maximum singular value.
         
    A low GR indicates that even when the inputs are noisy versions of one another,
    the reservoir maps them to very similar states.

    Parameters:
      states : np.ndarray
          A 2D numpy array containing reservoir state vectors.
          Accepted shapes:
             - (m, n): each row is a state vector (this will be transposed so that M has shape (n, m)), or
             - (n, m): each column is a state vector.
      threshold : float, optional
          A fraction (e.g. 1e-3) of the maximum singular value. Singular values below this threshold
          are considered negligible. Default is 1e-3.
    """
    def __init__(self, states: np.ndarray, threshold: float = 1e-3) -> None:
        self.threshold = threshold

        # Ensure states is a 2D array.
        if states.ndim != 2:
            raise ValueError("states must be a 2D numpy array.")

        m, n = states.shape
        self.M = states

    def compute_generalization_rank(self) -> Tuple[int, np.ndarray]:
        """
        Computes the effective (generalization) rank of matrix M via SVD.
        
        Returns:
          effective_rank : int
              The number of singular values sᵢ satisfying sᵢ > (threshold × s_max).
          singular_values : np.ndarray
              The singular values computed from the SVD.
        """
        # Compute the singular value decomposition of M.
        U, s, Vh = np.linalg.svd(self.M, full_matrices=False)
        s_max = np.max(s)
        effective_rank = np.sum(s > (self.threshold * s_max))
        return effective_rank, s

    def run_evaluation(self) -> Dict[str, Any]:
        """
        Runs the generalization rank evaluation.
        
        Returns:
          dict: A dictionary containing:
              - 'generalization_rank': The computed effective rank.
              - 'singular_values': The singular values of M.
              - 'M_shape': The shape of the matrix M.
              - 'threshold': The threshold used.
        """
        effective_rank, singular_values = self.compute_generalization_rank()
        logger.info(f"Computed Generalization Rank: {effective_rank}")
        return {
            'generalization_rank': effective_rank,
            'singular_values': singular_values,
            'M_shape': self.M.shape,
            'threshold': self.threshold
        }

