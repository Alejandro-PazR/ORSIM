import logging
from utils.log_format import CustomFormatter
from datetime import timedelta

import numpy as np
from scipy.integrate import solve_ivp

import core
from propagator import BasePropagator
from core.time import Epoch
from phenomena import Sun, Moon
from constants.general import Earth, Sun as Solar, Moon as Lunar

from propagator.perturbations.non_sphericity import asphericalAcceleration
from propagator.perturbations.third_body import thirdBodyAcceleration, ephemerisInterpolation
from propagator.perturbations.drag import dragAcceleration
from propagator.perturbations.radiation import (
	solarRadiationAcceleration,
	albedoAcceleration,
	longwaveAcceleration
)
from radiation import earthElements


# Logging settings
logger = logging.getLogger("Cowell")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)


class Cowell(BasePropagator):
	@classmethod
	def propagate(cls,
	    	spacecraft: core.spacecraft.SpaceCraft,
			state: core.state.StateVector,
			epoch: core.time.Epoch, tStop,
			*args
		):
		orbit = cls(spacecraft, state, epoch, tStop)
		u0 = orbit._state() # Initial state

		if not args:
			perturbations = (False, False, False, False, False, False)
		else:
			perturbations = args[0]

		gravity, drag, third_body, solar_radiation_pressure, albedo_pressure, longwave_pressure = perturbations

		if third_body or solar_radiation_pressure or albedo_pressure or longwave_pressure:
			ephemerisSun = ephemerisInterpolation(Sun, orbit._epoch, tStop, points=100)
			ephemerisMoon = ()
			if third_body:
				ephemerisMoon = ephemerisInterpolation(Moon, orbit._epoch, tStop, points=200)
		else:
			ephemerisSun, ephemerisMoon = (), ()

		if albedo_pressure or longwave_pressure:
			elements = earthElements(spacing=10)
		else:
			elements = ()

		stop_event = lithobrakeEvent()

		result = solve_ivp(
			orbit.diff_eq_twobody,
			(0, orbit._tStop),
			u0,
			args=(perturbations, ephemerisSun, ephemerisMoon, elements),
			events=stop_event,
			method='DOP853',
			dense_output=True,
			rtol=1e-11,
			atol=1e-12,
		)

		if not result.success:
			raise RuntimeError("Integration failed")
		
		if stop_event.flag:
			logger.warning(f"Lithobrake event detected at t = {stop_event.last_t} (s)")

		return result
	
	def diff_eq_twobody(self, t, u_, perturbations, ephemerisSun, ephemerisMoon, elements):
		x, y, z, vx, vy, vz = u_

		# ECI, GCRF vectors
		r = np.array([x, y, z])
		v = np.array([vx, vy, vz])
		r3 = np.linalg.norm(r)**3
		a = -(Earth.mu/r3) * r

		gravity, drag, third_body, solar_radiation_pressure, albedo_pressure, longwave_pressure = perturbations

		if third_body or solar_radiation_pressure or albedo_pressure or longwave_pressure:
			r_sun = ephemerisSun(t)
			if third_body:
				r_moon = ephemerisMoon(t)

		if albedo_pressure or longwave_pressure:
			new = self._epoch.date + timedelta(seconds=t)
			new_epoch = Epoch(new.year, new.month, new.day, new.hour, new.minute, new.second + new.microsecond*1e-6)

		if gravity:
			a += asphericalAcceleration(r)
		if drag:
			a += dragAcceleration(r, v, CD=self._spacecraft.CD, A=self._spacecraft.A_drag, m=self._spacecraft.mass)
		if third_body:
			a += thirdBodyAcceleration(Solar, r_sun, r)
			a += thirdBodyAcceleration(Lunar, r_moon, r)
		if solar_radiation_pressure:
			a += solarRadiationAcceleration(r_sun, r, CR=self._spacecraft.CR, A=self._spacecraft.A_srp, m=self._spacecraft.mass)
		if albedo_pressure:
			a += albedoAcceleration(new_epoch, elements, r_sun, r, CR=self._spacecraft.CR, A=self._spacecraft.A_srp, m=self._spacecraft.mass)
		if longwave_pressure:
			a += longwaveAcceleration(new_epoch, elements, r_sun, r, CR=self._spacecraft.CR, A=self._spacecraft.A_srp, m=self._spacecraft.mass)

		du = np.concatenate((v, a))
		return du


class lithobrakeEvent:
	def __init__(self, alt=10, terminal=True, direction=-1):
		self._terminal = terminal
		self._direction = direction
		self._last_t = None
		self._alt = alt
		self._flag = False

	def __call__(self, t, u, *args):
		self._last_t = t
		r_norm = np.linalg.norm(u[:3])

		event_call = r_norm - Earth.radius - self._alt
		if event_call < self._alt:
			self._flag = True
		# If this goes from +ve to -ve, altitude is decreasing.
		return (r_norm - Earth.radius - self._alt)

	@property
	def flag(self):
		return self._flag

	@property
	def terminal(self):
		return self._terminal

	@property
	def direction(self):
		return self._direction

	@property
	def last_t(self):
		return self._last_t