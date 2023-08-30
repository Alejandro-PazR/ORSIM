from numba import jit
import numpy as np

from _math.ellipsoid import areaEllipsoid
from _math.trigonometry import angle
from _math.vector import normalize
from constants.general import AU
from phenomena import shadowFunction, sight
from transformations.coordinates import ECEF_to_latlon, latlon_to_ECEF, gc_to_gd
from reduction.fk5 import FK5


def solarRadiationPressure(r_sun=149597870700.0) -> float:
	# L_sun = 3.828e26 # Solar luminance [W]
	# SF = L_sun / (4*np.pi*r_sun^2) # Solar irradiance [W/m^2]
	
	r_AU = r_sun/AU # Sun distance in [AU]

	SF = 1361 / r_AU**2 # Solar irradiance [W/m^2]
	c = 299792458 # Speed of light [m/s]
	p_srp = SF / c  # [N/m^2]
	return p_srp


def irradiance(epoch, elements, r_sun, r_sat):
	c = 299792458 # Speed of light [m/s]

	nu = shadowFunction(r_sun, r_sat)

	r_sunSat = r_sat - r_sun # [km]
	mod_r_sunSat = np.linalg.norm(r_sunSat)
	p_srp = solarRadiationPressure(mod_r_sunSat)
	
	solarIrradiation = nu * p_srp * c

	# ECEF Coordinates
	reduction = FK5(epoch)
	r_sat_ecef, _, _ = reduction.GCRF_to_ITRF(r_sat)
	r_sun_ecef, _, _ = reduction.GCRF_to_ITRF(r_sun)

	t0 = 2444960.5 # Base epoch JD UTC: December 22, 1981
	omega = 2*np.pi/365.25
	jd = epoch.JD_UTC

	albedoIrradiation = 0.0
	visibleEarthElementsIluminated = earthElementVisibility(elements, r_sat_ecef, r_sun_ecef)
	for element in visibleEarthElementsIluminated:
		area = element[0]
		phi = element[1]
		r_center = element[2:]

		r_elem_sat = r_sat_ecef - r_center
		r_elem_sun = r_sun_ecef - r_center

		# Unit vector
		mod_r = np.linalg.norm(r_sat_ecef - r_center)

		alpha = angle(r_elem_sat, r_center)
		theta_s = angle(r_elem_sun, r_center)

		alb = 0.34 + (0.1*np.cos(omega*(jd - t0))) * np.cos(phi) + 0.29*np.sin(phi)

		albedoIrradiation += (nu * alb * p_srp * c * area * np.cos(alpha) * np.cos(theta_s)) / (np.pi * mod_r**2)

	longwaveIrradiation = 0.0
	visibleEarthElements = earthElementVisibility(elements, r_sat_ecef)
	for element in visibleEarthElements:
		area = element[0]
		phi = element[1]
		r_center = element[2:]

		r_elem_sat = r_sat_ecef - r_center

		# Unit vector
		mod_r = np.linalg.norm(r_sat_ecef - r_center)

		alpha = angle(r_elem_sat, r_center)

		emis = 0.68 + (-0.07*np.cos(omega*(jd - t0))) * np.cos(phi) - 0.18*np.sin(phi)
		longwaveIrradiation += (0.25 * emis * p_srp * c * area * np.cos(alpha)) / (np.pi * mod_r**2)

	return solarIrradiation,  albedoIrradiation, longwaveIrradiation


def solarIrradiance(nu, r_sun, r_sat):
	nu = shadowFunction(r_sun, r_sat)
	
	c = 299792458 # Speed of light [m/s]

	r_sunSat = r_sat - r_sun # [km]
	mod_r_sunSat = np.linalg.norm(r_sunSat)
	p_srp = solarRadiationPressure(mod_r_sunSat)
	
	solarIrradiance = nu * p_srp * c

	return solarIrradiance


def albedoIrradiance(epoch, elements, r_sun, r_sat):
	# ECEF Coordinates
	reduction = FK5(epoch)
	r_sat_ecef, _, _ = reduction.GCRF_to_ITRF(r_sat)
	r_sun_ecef, _, _ = reduction.GCRF_to_ITRF(r_sun)

	t0 = 2444960.5 # Base epoch JD UTC: December 22, 1981
	omega = 2*np.pi/365.25
	jd = epoch.JD_UTC
	
	c = 299792458 # Speed of light [m/s]

	r_sunSat = r_sat - r_sun # [km]
	mod_r_sunSat = np.linalg.norm(r_sunSat)
	p_srp = solarRadiationPressure(mod_r_sunSat)

	albedoIrradiation = 0.0

	visibleEarthElements = earthElementVisibility(elements, r_sat_ecef, r_sun_ecef)

	for element in visibleEarthElements:
		area = element[0]
		phi = element[1]
		r_center = element[2:]

		r_elem_sat = r_sat_ecef - r_center
		r_elem_sun = r_sun_ecef - r_center

		# Unit vector
		mod_r = np.linalg.norm(r_sat_ecef - r_center)

		alpha = angle(r_elem_sat, r_center)
		theta_s = angle(r_elem_sun, r_center)
		alb = 0.34 + (0.1*np.cos(omega*(jd - t0))) * np.cos(phi) + 0.29*np.sin(phi)

		albedoIrradiation += (alb * p_srp * c * area * np.cos(alpha) * np.cos(theta_s)) / (np.pi * mod_r**2)

	return albedoIrradiation


def longwaveIrradiance(epoch, elements, r_sun, r_sat):
	# ECEF Coordinates
	reduction = FK5(epoch)
	r_sat_ecef, _, _ = reduction.GCRF_to_ITRF(r_sat)
	
	t0 = 2444960.5 # Base epoch JD UTC: December 22, 1981
	omega = 2*np.pi/365.25
	jd = epoch.JD_UTC
	
	c = 299792458 # Speed of light [m/s]

	r_sunSat = r_sat - r_sun # [km]
	mod_r_sunSat = np.linalg.norm(r_sunSat)
	p_srp = solarRadiationPressure(mod_r_sunSat)

	longwaveIrradiation = 0.0

	visibleEarthElements = earthElementVisibility(elements, r_sat_ecef)

	for element in visibleEarthElements:
		area = element[0]
		phi = element[1]
		r_center = element[2:]

		r_elem_sat = r_sat_ecef - r_center

		# Unit vector
		mod_r = np.linalg.norm(r_sat_ecef - r_center)

		alpha = angle(r_elem_sat, r_center)

		emis = 0.68 + (-0.07*np.cos(omega*(jd - t0))) * np.cos(phi) - 0.18*np.sin(phi)
		longwaveIrradiation += (0.25 * emis * p_srp * c * area * np.cos(alpha)) / (np.pi * mod_r**2)

	return longwaveIrradiation


def earthElements(spacing: int = 10):
	"""Center vector is in ECEF coordinates."""

	bound_lat = gc_to_gd(np.radians(np.linspace(90, -90+spacing, 180//spacing)))

	center_lat = gc_to_gd(np.radians(np.linspace(90-spacing/2, -90+spacing/2, 180//spacing)))
	center_lon = np.radians(np.linspace(0+spacing/2, 360-spacing/2, 360//spacing))

	areas, _ = np.meshgrid(center_lat, center_lon, indexing='ij')

	for i, lat in enumerate(bound_lat):
		for j in range(0, 360//spacing):
			spacingRad = np.radians(spacing)
			areas[i, j] = np.abs(areaEllipsoid(0, spacingRad, lat, lat-spacingRad))

	areas = np.expand_dims(areas, axis=2)
	elements = np.empty((2*90//spacing, 360//spacing, 5))

	for i, phi in enumerate(center_lat):
		for j, lamb in enumerate(center_lon):
			r = latlon_to_ECEF(phi, lamb, 0)
			elements[i, j, 0] = areas[i, j]
			elements[i, j, 1] = phi
			elements[i, j, 2:] = r

	return elements


def earthElementVisibility(elements, r_sat, r_sun=None):
	visibleElements = []

	for i in range(elements.shape[0]):
		for j in range(elements.shape[1]):
			r_center = elements[i, j, 2:]

			if r_sun is None:
				if sight(r_center, r_sat):
					visibleElements.append(elements[i, j, :])
			else:
				if sight(r_center, r_sat) and sight(r_center, r_sun):
					visibleElements.append(elements[i, j, :])

	return np.array(visibleElements)