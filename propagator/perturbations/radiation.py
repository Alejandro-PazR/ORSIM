from numba import jit
import numpy as np

from _math.ellipsoid import areaEllipsoid
from _math.trigonometry import angle
from _math.vector import normalize
from constants.general import AU
from phenomena import shadowFunction, sight
from transformations.coordinates import ECEF_to_latlon, latlon_to_ECEF, gc_to_gd
from reduction.fk5 import FK5
from radiation import (
	solarRadiationPressure,
	earthElementVisibility

)


def solarRadiationAcceleration(r_sun, r, CR, A, m):
	nu = shadowFunction(r_sun, r)

	r_sunSat = r - r_sun # [km]
	mod_r_sunSat = np.linalg.norm(r_sunSat)
	
	p_srp = solarRadiationPressure(mod_r_sunSat)

	conversion_factor = 0.001 # [m] -> [km]
	a = nu * p_srp * CR * (A/m) * (r_sunSat/mod_r_sunSat) * conversion_factor
	return a


def albedoAcceleration(epoch, elements, r_sun, r_sat, CR, A, m):
	# ECEF Coordinates
	reduction = FK5(epoch)
	r_sat_ecef, _, _ = reduction.GCRF_to_ITRF(r_sat)
	r_sun_ecef, _, _ = reduction.GCRF_to_ITRF(r_sun)

	t0 = 2444960.5 # Base epoch JD UTC: December 22, 1981
	omega = 2*np.pi/365.25
	jd = epoch.JD_UTC
	
	r_sunSat = r_sat - r_sun # [km]
	mod_r_sunSat = np.linalg.norm(r_sunSat)
	p_srp = solarRadiationPressure(mod_r_sunSat)

	grid_component_sum = 0.0

	visibleEarthElements = earthElementVisibility(elements, r_sat_ecef, r_sun_ecef)

	for element in visibleEarthElements:
		area = element[0]
		phi = element[1]
		r_center = element[2:]

		r_elem_sat = r_sat_ecef - r_center
		r_elem_sun = r_sun_ecef - r_center

		# Unit vector
		r, _, _ = reduction.ITRF_to_GCRF(r_ITRF=normalize(r_elem_sat))
		mod_r = np.linalg.norm(r_sat_ecef - r_center)

		alpha = angle(r_elem_sat, r_center)
		theta_s = angle(r_elem_sun, r_center)
		alb = 0.34 + (0.1*np.cos(omega*(jd - t0))) * np.cos(phi) + 0.29*np.sin(phi)

		grid_component_sum += alb * p_srp * np.cos(theta_s) * (A/(m*np.pi*mod_r**2)) * np.cos(alpha) * area * r

	conversion_factor = 0.001 # [m] -> [km]
	a = CR * grid_component_sum * conversion_factor
	return a


def longwaveAcceleration(epoch, elements, r_sun, r_sat, CR, A, m):
	# ECEF Coordinates
	reduction = FK5(epoch)
	r_sat_ecef, _, _ = reduction.GCRF_to_ITRF(r_sat)
	
	t0 = 2444960.5 # Base epoch JD UTC: December 22, 1981
	omega = 2*np.pi/365.25
	jd = epoch.JD_UTC
	
	r_sunSat = r_sat - r_sun # [km]
	mod_r_sunSat = np.linalg.norm(r_sunSat)
	p_srp = solarRadiationPressure(mod_r_sunSat)

	grid_component_sum = 0.0

	visibleEarthElements = earthElementVisibility(elements, r_sat_ecef)

	for element in visibleEarthElements:
		area = element[0]
		phi = element[1]
		r_center = element[2:]

		r_elem_sat = r_sat_ecef - r_center

		# Unit vector
		r, _, _ = reduction.ITRF_to_GCRF(r_ITRF=normalize(r_elem_sat))
		mod_r = np.linalg.norm(r_sat_ecef - r_center)

		alpha = angle(r_elem_sat, r_center)

		emis = 0.68 + (-0.07*np.cos(omega*(jd - t0))) * np.cos(phi) - 0.18*np.sin(phi)
		grid_component_sum += 0.25 * emis * p_srp * (A/(m*np.pi*mod_r**2)) * np.cos(alpha) * area * r

	conversion_factor = 0.001 # [m/s^2] -> [km/s^2]
	a = CR * grid_component_sum * conversion_factor
	return a