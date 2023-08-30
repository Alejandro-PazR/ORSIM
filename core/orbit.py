import time
from datetime import timedelta

import numpy as np

import core
from core.time import Epoch
from core.state import StateVector
from constants.general import Earth
from phenomena import Sun
from propagator.perturbations.third_body import ephemerisInterpolation
from radiation import (
	irradiance,
	earthElements
)
from plotter.orbit_plotter import orbitPlot

from propagator.cowell import Cowell
from propagator.kepler import KeplerUniversal

METHODS = {'Cowell': Cowell,
		   'KeplerUniversal': KeplerUniversal}


def sunSynchronousParams(sma, ecc=0.0):
	nodal_drift = 2*np.pi/(365.2421897*86400) # [rad/day]

	inc = np.arccos((-2*(sma**(7/2))*nodal_drift*((1-ecc**2)**2))/(3*(Earth.radius**2)*Earth.J2*np.sqrt(Earth.mu)))
	return inc


class OrbitCreation:
	def __init__(self, *_, **__):
		pass

	@classmethod
	def circular(cls):
		raise NotImplementedError
	
	@classmethod
	def sunSynchronous(cls,
			epoch: core.time.Epoch,
			spacecraft: core.spacecraft.SpaceCraft,	/,
			sma,
			ecc,
			raan,
			aop,
			ta,
		):
		if sma < Earth.radius:
			raise ValueError('Semimajor axis cannot be smaller than Earth\'s radius.')
		inc = sunSynchronousParams(sma, ecc)
		state = StateVector.from_elements(sma, ecc, inc, raan, aop, ta)
		return cls(epoch, spacecraft, state)

	@classmethod
	def from_elements(cls,
			epoch: core.time.Epoch,
			spacecraft: core.spacecraft.SpaceCraft, /,
			sma,
			ecc,
			inc,
			raan,
			aop,
			ta,
		):
		if sma < Earth.radius:
			raise ValueError('Semimajor axis cannot be smaller than Earth\'s radius.')
		state = StateVector.from_elements(sma, ecc, inc, raan, aop, ta)
		return cls(epoch, spacecraft, state)

	@classmethod
	def from_vectors(cls,
			epoch: core.time.Epoch,
			spacecraft: core.spacecraft.SpaceCraft, /,
			r: np.ndarray,
			v: np.ndarray,
		):
		state = StateVector.from_vectors(r, v)
		return cls(epoch, spacecraft, state)


class Orbit(OrbitCreation):
	def __init__(self,
			epoch: core.time.Epoch,
			spacecraft: core.spacecraft.SpaceCraft,
			state: core.state.StateVector,
		):
		self._epoch = epoch
		self._sc = spacecraft
		self._sv = state
		self._trajectory = False
		self.setPerturbations()

	def setPerturbations(self,
			gravity: bool = False,
			drag: bool = False,
			third_body: bool = False,
			solar_radiation_pressure: bool = False,
			albedo: bool = False,
			longwave: bool = False,
		):
		self._perturbations = (gravity, drag, third_body, solar_radiation_pressure, albedo, longwave)

	def propagate(self, tStop, method):

		if method not in METHODS:
			raise ValueError("`method` must be one of {}.".format(METHODS))

		if method in METHODS:
			method = METHODS[method]

		print(73*'=')
		print(f"Orbit propagation at {self._epoch}")
		print(73*'-')

		start = time.perf_counter()
		solution = method.propagate(self._sc, self._sv, self._epoch, tStop, self._perturbations)
		end = time.perf_counter()

		self.sol = solution
		
		if issubclass(method, Cowell):
			self._trajectory = True

			final = StateVector.from_vectors(
				np.array([self.sol.y[0, -1], self.sol.y[1, -1], self.sol.y[2, -1]]),
				np.array([self.sol.y[3, -1], self.sol.y[4, -1], self.sol.y[5, -1]])
			)
		else:
			final = self.sol

		new = self._epoch.date + timedelta(seconds=self.sol.t[-1])
		new_epoch = Epoch(new.year, new.month, new.day, new.hour, new.minute, new.second + new.microsecond*1e-6)

		messages = [
			f"Integration time (sec): {(end-start):.6f}",
			f"Final epoch: {new_epoch.date.strftime('%d-%B-%Y, %H:%M:%S')} UTC",
			f"Final position (km): [{final.position.x:.9f}, {final.position.y:.9f}, {final.position.z:.9f}]",
			f"Final velocity (km/s): [{final.velocity.x:.9f}, {final.velocity.y:.9f}, {final.velocity.z:.9f}]",
			"Final elements:",
			f"\t  SMA (km): {final.sma}",
			f"\t       ECC: {final.ecc}",
			f"\t INC (deg): {np.degrees(final.inc)%180}",
			f"\tRAAN (deg): {np.degrees(final.raan)%360}",
			f"\t AOP (deg): {np.degrees(final.aop)%360}",
			f"\t  TA (deg): {np.degrees(final.ta)%360}",
		]

		print(*messages[0:], sep = "\n")
		print(73*'=')
		
		return self.sol
	
	def radiation(self, gridSpacing=5):

		if not self._trajectory:
			raise ValueError("No propagated or continuous trajectory has been found.")

		print(73*'*')
		print("Received irradiation")
		print(73*'-')

		times = self.sol.t
		x = self.sol.y[0]
		y = self.sol.y[1]
		z = self.sol.y[2]
		trajectoryList = np.column_stack((x, y, z))

		tStop = times[-1]
		ephemerisSun = ephemerisInterpolation(Sun, self._epoch, tStop, points=int(2e-4*tStop))

		elements = earthElements(spacing=gridSpacing)

		solarIrradiance = []
		albedoIrradiance = []
		longwaveIrradiance = []

		start = time.perf_counter()

		for i, t in enumerate(times):
			new = self._epoch.date + timedelta(seconds=t)
			new_epoch = Epoch(new.year, new.month, new.day, new.hour, new.minute, new.second + new.microsecond*1e-6)
			
			r_sun = ephemerisSun(t)
			r = trajectoryList[i]
			
			sr, alb, olr = irradiance(new_epoch, elements, r_sun, r)
			solarIrradiance.append(sr)
			albedoIrradiance.append(alb)
			longwaveIrradiance.append(olr)

		end = time.perf_counter()

		print(f'Computation time (sec): {(end-start):.6f}')
		print(73*'*')

		return solarIrradiance, albedoIrradiance, longwaveIrradiance
	
	def plotTrajectory(self, daynight=True):
		if not self._trajectory:
			raise ValueError("No propagated or continuous trajectory has been found.")

		new = self._epoch.date + timedelta(seconds=self.sol.t[-1])
		new_epoch = Epoch(new.year, new.month, new.day, new.hour, new.minute, new.second + new.microsecond*1e-6)
		
		points = np.column_stack((self.sol.y[0], self.sol.y[1], self.sol.y[2]))

		orbitPlot(new_epoch, points, daynight)

	@property
	def initialState(self):
		return self._sv