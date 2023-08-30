import numpy as np

from constants.general import Earth
import core
from core.state import StateVector
from propagator import BasePropagator


def c2c3(psi):
	"""Find c2 and c3 for the Kepler's equation. (Vallado Algorithm 1)."""

	if psi > 1e-6:
		c2 = (1 - np.cos(np.sqrt(psi)))/psi
		c3 = (np.sqrt(psi) - np.sin(np.sqrt(psi)))/np.sqrt(psi**3)
	else:
		if psi < -1e-6:
			c2 = (1 - np.cosh(np.sqrt(-psi)))/psi
			c3 = (np.sinh(np.sqrt(-psi)) - np.sqrt(-psi))/np.sqrt((-psi)**3)
		else:
			c2 = 1/2
			c3 = 1/6

	return c2, c3


def keplerUniversal(r0_vec, v0_vec, delta_t):
	"""Kepler's problem propagator using universal formulation.

	Notes
	-----
	Vallado, algorithm 8.
	Newton-Raphson iteration.

	Parameters
	----------
	r0_vec : np.array
		Position vector.
	v0_vec : np.array
		Velocity vector.
	delta_t : float
		Time period in [s].

	Returns
	-------
	tuple, np.array
		r_vec : Position vector.
		v_vec : Velocity vector.
	"""

	r0 = np.linalg.norm(r0_vec)
	v0 = np.linalg.norm(v0_vec)

	xi = (v0**2)/2 - Earth.mu/r0
	
	a = -Earth.mu/(2*xi)
	alpha = 1/a

	if alpha > 1e-6: # Circle or ellipse
		x0 = np.sqrt(Earth.mu)*(delta_t)*alpha
		if alpha == 1:
			raise ValueError("Kepler: First guess too close to converge.")
	elif np.abs(alpha) < 1e-6: # Parabola
		h = np.linalg.norm(np.cross(r0_vec, v0_vec))
		p = h**2 / Earth.mu
		s = 0.5 * np.arctan(1/(3 * np.sqrt(Earth.mu/p**3) * delta_t))
		w = np.arctan(np.cbrt(np.tan(s)))
		x0 = np.sqrt(p) * 2 * 1/(np.tan(2 * w))
	elif alpha < -1e-6: # Hyperbola
		a = 1/alpha

		x0 = np.sign(delta_t) * np.sqrt(-a) * np.log((-2*Earth.mu*alpha*delta_t)/ \
			(np.dot(r0_vec, v0_vec) + np.sign(delta_t)*np.sqrt(-Earth.mu*a)*(1 - r0*alpha)))

	xn = x0
	while True:
		psi = xn**2 * alpha

		c2, c3 = c2c3(psi)

		r = xn**2*c2 + ((r0_vec @ v0_vec)*xn*(1 - psi*c3))/np.sqrt(Earth.mu) + r0*(1-psi*c2)
		xnext = xn + (np.sqrt(Earth.mu)*delta_t - xn**3*c3 - ((r0_vec @ v0_vec)/np.sqrt(Earth.mu))*xn**2*c2 - r0*xn*(1 - psi*c3))/r

		if np.abs(xnext - xn) < 1e-6:
			break
		else:
			xn = xnext

	f = 1 - (xn**2*c2)/r0
	g = delta_t - (xn**3*c3)/np.sqrt(Earth.mu)

	g_dot = 1 - (xn**2*c2)/r
	f_dot = np.sqrt(Earth.mu)*xn*(psi*c3 - 1)/(r*r0)

	r_vec = f*r0_vec + g*v0_vec
	v_vec = f_dot*r0_vec + g_dot*v0_vec

	if np.abs((f*g_dot - f_dot*g) - 1) > 1e5:
		raise RuntimeError("Kepler: propagator hasn't converged.")

	return r_vec, v_vec


class KeplerUniversal(BasePropagator):
	@classmethod
	def propagate(cls,
	    	spacecraft: core.spacecraft.SpaceCraft,
			state: core.state.StateVector,
			epoch: core.time.Epoch,
			tStop,
			*args
		):
		orbit = cls(spacecraft, state, epoch, tStop)

		r0 = orbit._state.position()
		v0 = orbit._state.velocity()

		r, v = keplerUniversal(r0, v0, tStop)
		solution = StateVector.from_vectors(r, v)

		return solution