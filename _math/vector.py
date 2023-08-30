from numba import jit
import numpy as np


@jit
def normalize(vec: np.ndarray) -> np.ndarray:
	"""Return a normalized unit vector."""
	return vec / np.linalg.norm(vec)