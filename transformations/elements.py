import numpy as np

from constants.general import Earth
from transformations.rotations import R1, R2, R3


def nu2anomaly(e, nu) -> float:
	"""Get various anomalies in term of the true anomaly.

	Parameters
	----------
	e : float
		Eccentricity.
	nu : float, [rad]
		True anomaly.

	Returns
	-------
	anomaly: float, [rad]
		Eccentric, parabolic or hyperbolic anomaly.
	"""

	if e < 1: # Elliptical orbit
		anomaly = np.arccos((e + np.cos(nu))/(1 + e*np.cos(nu)))
	elif e == 1: # Parabolic orbit
		anomaly = np.tan(nu/2)
	elif e < 1: # Hyperbolic orbit
		anomaly = np.arccosh((e + np.cos(nu))/(1 + e*np.cos(nu)))

	return anomaly


def anomaly2nu(e, anomaly, p=None, r=None):
	"""Get true anomaly from various anomalies.

	Parameters
	----------
	e : float
		Eccentricity.
	anomaly : float, [rad]
		Eccentric, parabolic or hyperbolic anomaly.
	p : int, optional
		Semiparameter, by default None
	r : int, optional
		Position vector magnitude, by default None.

	Returns
	-------
	nu
		True anomaly in [rad].
	"""

	if e < 1: # Elliptical orbit
		nu = np.arccos((np.cos(anomaly) - e)/(1 - e*np.cos(anomaly)))
	elif e == 1: # Parabolic orbit
		nu = np.arcsin(p*anomaly/r)
	elif e > 1: # Hyperbolic orbit 
		nu = np.arccosh((np.cosh(anomaly) - e)/(1 - e*np.cosh(anomaly)))

	return nu


def rv2coe(r, v) -> tuple:
	"""Convert position and velocity vectors to Kepler's orbital elements.

	Parameters
	----------
	r : numpy.ndarray
		Position vector
	v : numpy.ndarray
		Velocity vector

	Returns
	-------
	p : float
		Semiparameter.
	a : float
		Semimajor axis.
	e_mod : float
		Eccentricity.
	i : float
		Inclination, [rad].
	RAAN : float
		Right ascension of the ascending node, [rad].
	omega : float
		Argument of perigee, [rad].
	nu : float
		True anomaly, [rad].
	omega_true : float, if conditions apply
		True longitude of periapsis.
	u : float, if conditions apply
		Argument of latitude.
	lambda_true : float, if conditions apply
		True longitude.
	orbit_type : str
		'normal', 'elliptical equatorial', 'circular inclined', 'circular equatorial'
	"""

	h = np.cross(r, v)
	h_mod = np.linalg.norm(h)

	n = np.cross(np.array([0, 0, 1]), h)
	n_mod = np.linalg.norm(n)
	
	r_mod = np.linalg.norm(r)
	sq_v_mod = np.linalg.norm(v)**2
	
	e = ((sq_v_mod - Earth.mu/r_mod)*r - np.dot(r, v)*v)/Earth.mu
	e_mod = np.linalg.norm(e) # Eccentricity

	xi = sq_v_mod/2 - Earth.mu/r_mod

	if e_mod != 1.0:
		a = -Earth.mu/(2*xi) # Semimajor axis
		p = a * (1-e_mod**2) # Semiparameter
	else:
		p = h_mod**2/Earth.mu
		a = np.inf

	i = np.arccos(h[2]/h_mod) # Inclination

	raan = np.arccos(n[0]/n_mod) # Right ascension of the ascending node
	if n[1] < 0: raan = 2*np.pi - raan

	# Check for special cases

	if e_mod < 1e-5 and i < 1e-3: # Circular equatorial
		lambda_true = np.arccos(r[0]/r_mod) # True longitude
		if r[1] < 0: lambda_true = 2*np.pi - lambda_true
		return (p, a, e_mod, i, 0, 0, lambda_true, 'circular equatorial')
	
	elif e_mod < 1e-5: # Circular inclined
		u = np.arccos(np.dot(n, r)/(n_mod*r_mod)) # Argument of latitude
		if r[2] < 0: u = 2*np.pi - u
		return (p, a, e_mod, i, raan, 0, u, 'circular inclined')
	
	elif i < 1e-3: # Elliptical equatorial
		omega_true = np.arccos(e[0]/e_mod) # True longitude of periapsis
		if e[1] < 0: omega_true = 2*np.pi - omega_true
		return (p, a, e_mod, i, 0, 0, omega_true, 'elliptical equatorial')
	
	omega = np.arccos(np.dot(n, e)/(n_mod*e_mod)) # Argument of perigee
	if e[2] < 0: omega = 2*np.pi - omega

	nu = np.arccos(np.dot(e, r)/(e_mod*r_mod)) # True anomaly
	if np.dot(r, v) < 0: nu = 2*np.pi - nu

	return (p, a, e_mod, i, raan, omega, nu, 'normal')


def coe2rv(p, e, i, RAAN, omega, nu) -> tuple:
	"""Determine the position and velocity from orbital elements.

	Parameters
	----------
	p : float
		Semiparameter.
	e : float
		Eccentricity.
	i : float
		Inclination, [rad].
	RAAN : float
		Right ascension of the ascending node, [rad].
	omega : float
		Argument of perigee, [rad].
	nu : float
		True anomaly, [rad]. If conditions apply could be one of the following
			omega_true : if conditions apply
				True longitude of periapsis.
			u : if conditions apply
				Argument of latitude.
			lambda_true : if conditions apply
				True longitude.

	Returns
	-------
	r_IJK: numpy.ndarray
		Position vector in the geocentric equatorial coordinate system
	v_IJK: numpy.ndarray
		Velocity vector in the geocentric equatorial coordinate system
	"""

	# Check for special cases
	
	if i < 1e-3: # Elliptical equatorial
		omega = 0
		RAAN = 0
	elif e < 1e-5: # Circular inclined
		omega = 0
	elif e < 1e-5 and i < 1e-3: # Circular equatorial
		RAAN = 0
		omega = nu

	r_PQW = np.array([(p*np.cos(nu))/(1 + e*np.cos(nu)),
				  	  (p*np.sin(nu))/(1 + e*np.cos(nu)),
					   0])

	v_PQW = np.array([-np.sqrt(Earth.mu/p)*np.sin(nu),
		   			   np.sqrt(Earth.mu/p)*(e + np.cos(nu)),
				  	   0])
	
	r_IJK = R3(-RAAN) @ R1(-i) @ R3(-omega) @ r_PQW 
	v_IJK = R3(-RAAN) @ R1(-i) @ R3(-omega) @ v_PQW

	return r_IJK, v_IJK