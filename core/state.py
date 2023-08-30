from typing import Optional, Any
from dataclasses import dataclass
import numpy

from transformations.elements import coe2rv, rv2coe


@dataclass(repr=False)
class Vector:
	vec: numpy.ndarray

	def __call__(self):
		return self.vec
	
	def __repr__(self):
		return f'Vector(x={self.x}, y={self.y}, z={self.z})'

	@property
	def x(self):
		return self.vec[0]
	
	@property
	def y(self):
		return self.vec[1]
	
	@property
	def z(self):
		return self.vec[2]


class StateVector:
	def __init__(self, stateVector: numpy.ndarray):
		self._state = stateVector
		self._r = Vector(self._state[:3])
		self._v = Vector(self._state[3:])
		self._elements = rv2coe(self.position(), self.velocity())
				
	@classmethod
	def from_elements(cls,
			sma: float,
			ecc: float,
			inc: float,
			raan: float,
			aop: float,
			ta: float,
			arglat: Optional[float] = None,
			truelon: Optional[float] = None,
			lonper: Optional[float] = None
		):

		if ecc < 1:
			p = sma*(1-ecc**2)
		
		else:
			p = sma

		if arglat:
			r, v = coe2rv(p, ecc, inc, raan, aop, arglat)
		elif truelon:
			r, v = coe2rv(p, ecc, inc, raan, aop, truelon)
		elif lonper:
			r, v = coe2rv(p, ecc, inc, raan, aop, lonper)
		else:
			r, v = coe2rv(p, ecc, inc, raan, aop, ta)

		state = numpy.concatenate((r, v))
		return cls(state)

	@classmethod
	def from_vectors(cls,
			r: numpy.ndarray,
			v: numpy.ndarray,
		):
		state = numpy.concatenate((r, v))
		return cls(state)
	
	def __call__(self) -> numpy.ndarray:
		"""Return the state vector."""
		return self._state
	
	def __repr__(self) -> str:
		return f'r: {self.position}\nv: {self.velocity}'

	@property
	def position(self):
		"""Position vector."""
		return self._r
	
	@property
	def velocity(self):
		"""Velocity vector."""
		return self._v
	
	@property
	def sma(self):
		"""Semimajor axis."""
		return self._elements[1]
	
	@property
	def ecc(self):
		"""Eccentricity."""
		return self._elements[2]
	
	@property
	def inc(self):
		"""Inclination."""
		return self._elements[3]
	
	@property
	def raan(self):
		"""Right ascension of the ascending node."""
		return self._elements[4]
	
	@property
	def aop(self):
		"""Argument of perigee."""
		return self._elements[5]
	
	@property
	def ta(self):
		"""True anomaly."""
		# if self._elements[-1] != 'normal':
		# 	raise ValueError("Specialized orbit")
		return self._elements[6]
	
	@property
	def arglat(self):
		"""Argument of latitude for circular inclined orbits."""
		if self._elements[-1] != 'circular inclined':
			raise ValueError("The orbit is not circular inclined")
		else:
			return self._elements[6]

	@property
	def truelon(self):
		"""True longitude for circular equatorial orbits."""
		if self._elements[-1] != 'circular equatorial':
			raise ValueError("The orbit is not circular equatorial")
		else:
			return self._elements[6]

	@property
	def lonper(self):
		"""True longitude of periapsis for elliptical equatorial."""
		if self._elements[-1] != 'elliptical equatorial':
			raise ValueError("The orbit is not elliptical equatorial")
		else:
			return self._elements[6]

	def to_vectors(self) -> tuple:
		"""Returns a tuple with both vectors."""
		return self.position(), self.velocity()

	def to_elements(self) -> tuple:
		"""Returns a tuple with orbital elements."""
		element_tuple = (self.sma,
						 self.ecc,
						 self.inc,
						 self.raan,
						 self.aop,
						)

		match self._elements[-1]:
			case 'normal':
				element_tuple + self.ta
			case 'circular inclined':
				element_tuple + self.arglat
			case 'circular equatorial':
				element_tuple + self.truelon
			case 'elliptical equatorial':
				element_tuple + self.lonper

		return element_tuple

	def to_dict(self) -> dict:
		"""Returns a dictionary with orbital elements."""
		prev = ('sma', 'ecc', 'inc', 'raan', 'aop')

		match self._elements[-1]:
			case 'normal':
				keys = (*prev, 'ta')
			case 'circular inclined':
				keys = (*prev, 'arglat')
			case 'circular equatorial':
				keys = (*prev, 'truelon')
			case 'elliptical equatorial':
				keys = (*prev, 'lonper')

		return dict(zip(keys, self._elements[1:7]))