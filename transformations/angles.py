import numpy as np


def secToRad(ang):
	return np.radians(ang/3600)

def secToDeg(ang):
	return ang/3600

def degToRad(ang):
	return (np.degrees(ang))

def degToSec(ang):
	return ang*3600

def radToSec(ang):
	return np.degrees(ang)*3600

def radToDMS(ang):
	temp = ang*(180/np.pi)
	deg = np.trunc(temp)
	min = np.trunc((temp - deg)*60)
	sec = (temp - deg - min/60)*3600

	return (deg, min, sec)

def eulerAngles(matrix):
	X = matrix[0,:]
	Y = matrix[1,:]
	Z = matrix[2,:]
	
	alpha = np.arctan2(Z[0], -Z[1])
	beta = np.arccos(Z[2])
	gamma = np.arctan2(X[2], Y[2])

	return alpha, beta, gamma
