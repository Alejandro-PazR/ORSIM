from numba import jit
import numpy as np

from _math.trigonometry import *
from core.time import Epoch
from transformations.angles import *
from constants.general import Earth, AU
from constants.general import Sun as Solar
from reduction.fk5 import FK5


class Sun:
	def __init__(self, epoch):
		self.UTC = epoch.UTC
		T_UT1 = epoch.T_UT1
		T_TDB = epoch.T_TDB
		self.T_TT = epoch.T_TT

		lambda_M_sun = 280.460 + 36000.771*T_UT1 # Mean longitude of the Sun
		self.M_sun = 357.5291092 + 35999.05034*T_TDB # Mean anomaly of the Sun [deg]

		# Ecliptic longitude of the Sun [deg]
		self.lambda_ecliptic = lambda_M_sun + 1.914666471*sind(self.M_sun) + 0.019994643*sind(2*self.M_sun)

		# Magnitude of distance to the Sun [km]
		self.r_sun = AU * (1.000140612 - 0.016708617*cosd(self.M_sun) - 0.000139589*cosd(2*self.M_sun))
		
		# Obliquity of the ecliptic
		self.epsilon = 23.439291 - 0.0130042*T_TDB
	
	def vector(self) -> np.ndarray:
		"""Returns the Sun position vector in km in J2000 coordinates."""

		r_MOD = self.r_sun * np.array([cosd(self.lambda_ecliptic),
								cosd(self.epsilon) * sind(self.lambda_ecliptic),
								sind(self.epsilon) * sind(self.lambda_ecliptic)])
		
		return FK5.MOD_to_J2000(self.T_TT, r_MOD)
	
	@property
	def declination(self):
		"""Sun's declination in [rad]."""
		return np.arcsin(sind(self.epsilon)*sind(self.lambda_ecliptic))
	
	@property
	def rightAscension(self):
		"""Sun's right ascension in [rad]."""
		return np.arctan2(cosd(self.epsilon)*sind(self.lambda_ecliptic), cosd(self.lambda_ecliptic))
	
	@property
	def equation_of_time(self):
		"""Equation of time in [min]"""
		return 4*(-1.914666471*sind(self.M_sun) - 0.019994643*sind(2*self.M_sun)
			+ 2.466*sind(2*self.lambda_ecliptic) - 0.0053*sind(4*self.lambda_ecliptic))
	
	@property
	def subsolar(self):
		"""Subsolar point in [deg]."""
		lat_ss= np.degrees(self.declination)
		lon_ss = -(1/240)*(self.UTC - 12*3600 + (self.equation_of_time)*60)
		return lat_ss, lon_ss
	
	@property
	def antisolar(self):
		"""Antisolar point in [deg]."""
		lat_ss, lon_ss = self.subsolar
		lat_as = -lat_ss
		lon_as = -np.sign(lon_ss)*(180 - np.abs(lon_ss))
		return lat_as, lon_as


class Moon:
	def __init__(self, epoch):
		self.T_TDB = epoch.T_TDB
		self.T_TT = epoch.T_TT
		
		# Ecliptic longitude of the Moon
		self.lambda_ecliptic = (218.32 + 481267.8813*self.T_TDB + 6.29*sind(134.9 + 477198.85*self.T_TDB)
			- 1.27*sind(259.2 - 413335.38*self.T_TDB) + 0.66*sind(235.7 + 890534.23*self.T_TDB)
			+ 0.21*sind(269.9 + 954397.70*self.T_TDB) - 0.19*sind(357.5 + 35999.05*self.T_TDB)
			- 0.11*sind(186.6 + 966404.05*self.T_TDB))
		
		# Ecliptic latitude of the Moon
		self.phi_ecliptic = (5.13*sind(93.3 + 483202.03*self.T_TDB) + 0.28*sind(228.2 + 960400.87*self.T_TDB)
			- 0.28*sind(318.3 + 6003.18*self.T_TDB) - 0.17*sind(217.6 - 407332.20*self.T_TDB))
		
		# Horizontal parallax
		self.parallax = (0.9508 + 0.0518*cosd(134.9 + 477198.85*self.T_TDB)
			+ 0.0095*cosd(259.2 - 413335.38*self.T_TDB) + 0.0078*cosd(235.7 + 890534.23*self.T_TDB)
			+ 0.0028*cosd(269.9 + 954397.70*self.T_TDB))
		
		# Obliquity of the ecliptic
		self.epsilon = 23.439291 - 0.0130042*self.T_TDB - 1.64e-7*self.T_TDB**2 + 5.014e-7*self.T_TDB**3
		# Magnitude of distance to the Moon
		self.r_moon = Earth.radius * (1/sind(self.parallax))
	
	def vector(self) -> np.ndarray:
		"""Returns the Moon position vector in [km]."""
		r_MOD = self.r_moon * np.array([cosd(self.phi_ecliptic) * cosd(self.lambda_ecliptic),
			cosd(self.epsilon) * cosd(self.phi_ecliptic) * sind(self.lambda_ecliptic) - sind(self.epsilon) * sind(self.phi_ecliptic),
			sind(self.epsilon) * cosd(self.phi_ecliptic) * sind(self.lambda_ecliptic) + cosd(self.epsilon) * sind(self.phi_ecliptic)])

		return FK5.MOD_to_J2000(self.T_TT, r_MOD)

	@property
	def rightAscension(self):
		"""Moon's right ascension in [rad]."""
		return np.arctan2(cosd(self.epsilon) * cosd(self.phi_ecliptic) * sind(self.lambda_ecliptic) - sind(self.epsilon) * sind(self.phi_ecliptic),
		    cosd(self.phi_ecliptic) * cosd(self.lambda_ecliptic))

	@property
	def declination(self):
		"""Moon declination in [rad]."""
		return np.arcsin(sind(self.phi_ecliptic) * cosd(self.epsilon) + cosd(self.phi_ecliptic) * sind(self.epsilon) * sind(self.lambda_ecliptic))


@jit
def shadowFunction(r_sun, r_sat):
	"""
	Satellite Orbits Models, Methods and Applications (Oliver Montenbruck, Eberhard Gill)
	Section 3.4.2 Shadow Function, pag 81-82
	"""

	Solar_radius = 696000 # Solar.radius
	Earth_radius = 6378.1363 # Earth.radius

	a = np.arcsin(Solar_radius/np.linalg.norm(r_sun - r_sat))
	b = np.arcsin(Earth_radius/np.linalg.norm(r_sat))
	c = np.arccos((-r_sat @ (r_sun - r_sat))/(np.linalg.norm(r_sat) * np.linalg.norm(r_sun - r_sat)))

	if np.abs(a-b) < c < (a+b):
		x = (c**2 + a**2 - b**2)/(2*c)
		y = np.sqrt(a**2 - x**2)

		A = a**2 * np.arccos(x/a) + b**2 * np.arccos((c-x)/b) - c*y
		return 1 - (A)/(np.pi*a**2) # Penumbra (value between 0 and 1)
	elif c < np.abs(a-b) :
		return 0 # Umbra
	elif c >= (a+b):
		return 1 # Sunlight


@jit
def shadow(r_sun, r_sat):
	"""Vallado's implementation."""
	shadow = 'none'

	Solar_radius = 696000 # Solar.radius
	Earth_radius = 6378.1363 # Earth.radius

	ang_umbra = np.arcsin((Solar_radius - Earth_radius)/AU)
	ang_penumbra = np.arcsin((Solar_radius + Earth_radius)/AU)

	if np.dot(r_sun, r_sat) < 0:
		sigma = angle(-r_sun, r_sat)
		sat_horiz = np.linalg.norm(r_sat) * cos(sigma)
		sat_vert = np.linalg.norm(r_sat) * sin(sigma)
		x = Earth.radius/sin(ang_penumbra)
		pen_vert = tan(ang_penumbra) * (x + sat_horiz)

		if sat_vert <= pen_vert:
			shadow = 'penumbra'
			y = Earth.radius/sin(ang_umbra)
			umb_vert = tan(ang_umbra) * (y - sat_horiz)
			
			if sat_vert <= umb_vert:
				shadow = 'umbra'

	return shadow


@jit
def sight(r1, r2):
	"""Given two vectors determines if there is a line-of-sight between the two objects.

	We are assuming an oblate Earth by scaling the k component (Rapid determination
	of satellite visibility periods, S. Alfano, D. Negron, 1993)

	Parameters
	----------
	r1 : np.ndarray
	r2 : np.ndarray

	Returns
	-------
	bool
		True if there is LOS, False if there is not.
	"""
	# Earth ellipsoidal scaling as per Vallado eq. 1-5
	scale_factor = 1 / np.sqrt(1 - 0.081819221456**2)

	_r1, _r2 = r1.copy(), r2.copy()

	_r1[2] *= scale_factor
	_r2[2] *= scale_factor

	sqmag_1 = np.linalg.norm(_r1)**2
	sqmag_2 = np.linalg.norm(_r2)**2

	r1_dot_r2 = np.dot(_r1, _r2)

	if (sqmag_1 + sqmag_2 - 2*r1_dot_r2) < 0.0001:
		t_min = 0
	else:
		t_min = (sqmag_1 - r1_dot_r2)/(sqmag_1 + sqmag_2 - 2*r1_dot_r2)

	if t_min < 0 or t_min > 1:
		los = True
	else:
		if ((1 - t_min) * sqmag_1 + r1_dot_r2 * t_min) / 6378.1363**2 >= 1:
			los = True
		else:
			los = False

	return los