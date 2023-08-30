from dataclasses import dataclass

import numpy as np


@dataclass
class Planet:
	name: str
	mu: float # Gravitational parameter [km^3/s^2]
	radius: float # Equatorial radius [km]
	eccentricity: float # Oblate eccentricity
	J2: float # J2 perturbation
	J3: float = None # J3 perturbation
	J4: float = None # J4 perturbation
	J5: float = None # J5 perturbation
	J6: float = None # J6 perturbation

	def __eq__(self, __value: object) -> bool:
		return self.name == __value.name

	@property
	def reciprocal_flattening(self) -> float:
		return 1 - np.sqrt(1 - self.eccentricity**2) # f = 1 - sqrt(1 - e^2)


AU = 149597870.7 # Astronomical Unit [km]


Earth = Planet(
	name = 'Earth',
	mu = 398600.4415,
	radius = 6378.1363,
	eccentricity = 0.081819221456,
	J2 = 0.0010826261738522227,
	J3 = -2.5324105185677225e-06,
	J4 = -1.6198975999169731e-06,
	J5 = -2.2775359073083616e-07,
	J6 = 5.406665762838132e-07
)

Sun = Planet(
	name = 'Sun',
	mu = 1.32712428e11,
	radius = 696000,
	eccentricity = None,
	J2 = None
)

Moon = Planet(
	name = 'Moon',
	mu = 4902.799,
	radius = 1738.0,
	eccentricity = None,
	J2 = 0.0002027
)