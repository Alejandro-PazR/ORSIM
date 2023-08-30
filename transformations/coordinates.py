import numpy as np

from constants.general import Earth


def ECEF_to_latlon(r, tolerance=1e-8):
	rI = r[0]
	rJ = r[1]
	rKsat = r[2]

	rdsat = np.sqrt(rI**2 + rJ**2)
	alpha = np.arctan2(rJ, rI)
	lambda_sat = alpha

	if np.abs(lambda_sat) >= np.pi:
		if lambda_sat < 0:
			lambda_sat += 2*np.pi
		else:
			lambda_sat -= 2*np.pi
			
	delta = np.arcsin(rKsat/np.linalg.norm(r))

	phi_gd = delta
	rdelta = rdsat
	rK = rKsat

	while True:
		C = Earth.radius / np.sqrt(1 - Earth.eccentricity**2 * np.sin(phi_gd)**2)
		phi_gd_new = np.arctan((rK + C*Earth.eccentricity**2*np.sin(phi_gd))/rdelta)

		if np.abs(phi_gd_new - phi_gd) < tolerance:
			phi_gd = phi_gd_new
			break
		else:
			phi_gd = phi_gd_new
	
	if np.abs(np.pi/2 - phi_gd) < 2e-2: # If near the poles (~1º)
		S = C * (1 - Earth.eccentricity**2)
		h_ellip = rK / np.sin(phi_gd) - S
	else:
		h_ellip = rdelta/np.cos(phi_gd) - C

	return phi_gd, lambda_sat, h_ellip


def latlon_to_ECEF(phi_gd, lambd, h_ellip=0.):
	C = Earth.radius/(np.sqrt(1 - (Earth.eccentricity**2 * (np.sin(phi_gd))**2)))
	S = C * (1 - Earth.eccentricity**2)

	rdelta = (C + h_ellip)*np.cos(phi_gd)
	rK = (S + h_ellip)*np.sin(phi_gd)

	r_siteECEF = np.array([rdelta*np.cos(lambd), rdelta*np.sin(lambd), rK])
	return r_siteECEF


def gd_to_gc(phi_gd):
	"""Geodetic (ellipsoidal) latitude to geocentric (spherical) latitude."""
	return np.arctan((1 - Earth.eccentricity**2) * np.tan(phi_gd))


def gc_to_gd(phi_gc):
	"""Geocentric (spherical) latitude to geodetic (ellipsoidal) latitude."""
	return np.arctan(np.tan(phi_gc) / (1 - Earth.eccentricity**2))