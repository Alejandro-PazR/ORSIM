from numba import jit

import numpy as np
from numpy import sin, cos


def R1(alpha):
	return np.array([[1, 0, 0], 
					  [0, cos(alpha), sin(alpha)],
					  [0, -sin(alpha), cos(alpha)]])


def R2(alpha):
	return np.array([[cos(alpha), 0, -sin(alpha)], 
					  [0, 1, 0],
					  [sin(alpha), 0, cos(alpha)]])


def R3(alpha):
	return np.array([[cos(alpha), sin(alpha), 0], 
					  [-sin(alpha), cos(alpha), 0],
					  [0, 0, 1]])