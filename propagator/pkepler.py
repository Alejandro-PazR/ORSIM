import numpy as np

from constants.general import Earth
from transformations.elements import nu2anomaly, anomaly2nu, rv2coe, coe2rv


def keplerEquationE(M, ecc, tolerance=1e-8):
	"""Newton-Raphson solution of Kepler's Equation.

	Parameters
	----------
	M : float, [rad]
		Mean anomaly
	ecc : float
		Eccentricity
	tolerance : float, optional
		Tolerance setting, by default 1e-8

	Returns
	-------
	E : float, [rad]
		Eccentric anomaly
	"""

	if -np.pi < M < 0 or M > np.pi:
		E_0 = M - ecc
	else:
		E_0 = M + ecc

	E_n = E_0
	
	while True:
		E_next = E_n + (M - E_n + ecc*np.sin(E_n))/(1 - ecc*np.cos(E_n))

		if np.abs(E_next - E_n) < tolerance:
			return E_next
		else:
			E_n = E_next


def pKepler(r0, v0, dt, n0_dot, n0_ddot):
	"""Propagates position and velocity over a time period
		accounting for perturbations caused by J2.

	Parameters
	----------
	r0 : numpy.ndarray
		Position vector
	v0 : numpyp.ndarray
		Position vector
	dt : float
		Time period in [s]
	n0_dot : float
		Time rate of change of n0, [rad/sec]
	n0_ddot : float
		Time acceleration of change of n0, [rad/sec2]
	"""

	p0, a0, e0, i0, RAAN0, omega0, nu0, orbit_type = rv2coe(r0, v0)
	
	if e0 is not 0:
		E0 = nu2anomaly(nu0, e0)
	else:
		E0 = nu0
	
	M0 = E0 - e0*np.sin(E0)
	p0 = a0*(1 - e0**2)
	n0 = np.sqrt(Earth.mu/a0**3)

	# Update for perturbations
	a = a0 - (2*a0*n0_dot*dt) / (3*n0)
	e = e0 - (2*(1 - e0)*n0_dot*dt) / (3*n0)
	RAAN = RAAN0 - (3*n0*Earth.radius**2*Earth.J2*np.cos(i0)*dt) / (2*p0**2)
	omega = omega0 + (3*n0*Earth.radius**2*Earth.J2*(4 - 5*np.sin(i0)**2)*dt) / (4*p0**2)
	M = M0 + n0*dt + (n0_dot * dt**2)/2 + (n0_ddot * dt**3)/6
	p = a*(1 - e**2)
	E = keplerEquationE(M, e)

	if e != 0:
		nu = anomaly2nu(e, E)
	else: # Circular equatorial or circular inclined
		nu = E

	r, v = coe2rv(p, e, i0, RAAN, omega, nu)

	return r, v