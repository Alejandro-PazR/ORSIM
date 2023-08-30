from numba import jit
import numpy as np
from numpy import sin, cos, tan, radians, around


@jit
def sind(x):
	"""Sine function in degrees with 16 decimals of precision."""
	return around(sin(radians(x)), 16)


@jit
def cosd(x):
	"""Cosine function in degrees with 16 decimals of precision."""
	return around(cos(radians(x)), 16)


@jit
def angle(A, B):
	"""Find the angle between two vectors."""
	return np.arccos(np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B)))