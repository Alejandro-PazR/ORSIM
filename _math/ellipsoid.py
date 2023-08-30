from numba import jit
import numpy as np
from constants.general import Earth


@jit
def areaEllipsoid(lon1, lon2, lat1, lat2):
	e = 0.081819221456 # Earth.eccentricity
	a = 6378.1363 # Earth.radius
	b = a * (1 - 0.0033528131779361203) #  Reciprocal flattening, polar radius
	
	A_E = (b**2 / 2) * (lon2 - lon1) * (
		(np.sin(lat2)/(1-e**2*(np.sin(lat2))**2) + (1/(2*e)) * np.log((1+e*np.sin(lat2))/(1-e*np.sin(lat2))))
		- (np.sin(lat1)/(1-e**2*(np.sin(lat1))**2) + (1/(2*e)) * np.log((1+e*np.sin(lat1))/(1-e*np.sin(lat1))))
	)
	return A_E


@jit
def earthOrthodromic(lat_1, lat_2, lon_1, lon_2):
	lat1 = np.radians(lat_1)
	lat2 = np.radians(lat_2)
	delta_lat = np.radians(np.abs(lat_2 - lat_1))
	delta_lon = np.radians(np.abs(lon_2 - lon_1))

	# Haversine formula
	a = np.sin(delta_lat/2)*np.sin(delta_lat/2) + np.cos(lat1)*np.cos(lat2)*np.sin(delta_lon/2)*np.sin(delta_lon/2)
	c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

	return 6378.1363 * c