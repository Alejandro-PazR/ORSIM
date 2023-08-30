import numpy as np
import pandas as pd

from constants.general import Earth
from transformations.coordinates import ECEF_to_latlon
from data.SW import getSWdata
from phenomena.phenomena import Sun


def JachiaRoberts(epoch, r):

	# Same as exponential Density
	rho_std = exponentialDensity(r)

	phi_gd, _, h_ellp = ECEF_to_latlon(r)
	
	kp = getSWdata(epoch.date, f'KP{int(np.floor(epoch.date.hour/3))+1}')

	if h_ellp < 200:
		G = 0.012*kp + 1.2e-5*np.exp(kp)
	else:
		G = 0

	T1958 = 2436204/365.2422
	LT = 0.014 * (h_ellp - 90) * np.sin(2*np.pi*T1958 + 1.72) * np.sin(phi_gd) * np.abs(np.sin(phi_gd)) * np.exp(-0.0013*(h_ellp-90)**2)
	
	tau_sa = T1958 + 0.09544*((0.5 + 0.5*np.sin(2*np.pi*T1958 + 6.035)**1.65) - 0.5)
	SA = (5.876e-7*h_ellp**2.331 + 0.06328) * np.exp(-0.002868*h_ellp) * (0.02835 + (0.3817+0.17829*np.sin(2*np.pi*tau_sa+4.137))*np.sin(4*np.pi*tau_sa+4.259))

	corr = G + SA + LT
	rho = rho_std * 10**(corr)

	return rho

def exponentialDensity(r):
	# Assuming a perfect spherical Earth for speed
	h_ellp = np.linalg.norm(r) - Earth.radius

	h0_data = (1000, 900, 800, 700, 600, 500, 450, 400, 350, 300, 250, 200, 180, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 25, 0)
	#h0_data = (0, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 180, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000)
	rho0_data = (3.019e-15, 5.245e-15, 1.170e-14, 3.614e-14, 1.454e-13, 6.967e-13, 1.585e-12, 3.725e-12, 9.518e-12, 2.418e-11, 7.248e-11, 2.789e-10, 5.464e-10, 2.070e-9, 3.845e-9, 8.484e-9, 2.438e-8, 9.661e-8, 5.297e-7, 3.396e-6, 1.905e-5, 8.770e-5, 3.206e-4, 1.057e-3, 3.972e-3, 1.774e-2, 3.899e-2, 1.225)
	#rho0_data = (1.225, 3.899e-2, 1.774e-2, 3.972e-3, 1.057e-3, 3.206e-4, 8.770e-5, 1.905e-5, 3.396e-6, 5.297e-7, 9.661e-8, 2.438e-8, 8.484e-9, 3.845e-9, 2.070e-9, 5.464e-10, 2.789e-10, 7.248e-11, 2.418e-11, 9.518e-12, 3.725e-12, 1.585e-12, 6.967e-13, 1.454e-13, 3.614e-14, 1.170e-14, 5.245e-15, 3.019e-15)
	H_data = (268.00, 181.05, 124.64, 88.667, 71.835, 63.822, 60.828, 58.515, 53.298, 53.628, 45.546, 37.105, 29.740, 22.523, 16.149, 12.636, 9.473, 7.263, 5.877, 5.382, 5.799, 6.549, 7.714, 8.382, 7.554, 6.682, 6.349, 7.249)
	#H_data = (7.249, 6.349, 6.682, 7.554, 8.382, 7.714, 6.549, 5.799, 5.382, 5.877, 7.263, 9.473, 12.636, 16.149, 22.523, 29.740, 37.105, 45.546, 53.628, 53.298, 58.515, 60.828, 63.822, 71.835, 88.667, 124.64, 181.05, 268.00)
	
	for i, h in enumerate(h0_data):
		h0 = h0_data[i]
		rho0 = rho0_data[i]
		H = H_data[i]
		
		if h_ellp >= h:
			break

	rho_std = rho0 * np.exp((h0 - h_ellp)/H)
	return rho_std


def temperature(epoch, altitude):
	F107_daily = getSWdata(epoch.date, 'F10.7_OBS')
	F107_81 = getSWdata(epoch.date, 'F10.7_OBS_CENTER81')

	# Nightime global exospheric temperature, Tc [K]
	Tc = 379 + 3.24*F107_81 + 1.3*(F107_daily - F107_81)

	declination = Sun(epoch).declination
	phi_gd = None
	LHA = None
	hellp = None

	eta = np.abs(phi_gd - declination)/2
	theta = np.abs(phi_gd + declination)/2

	tau = LHA - 37 + 6*np.sin(LHA + 43)

	# Uncorrected exospheric temperatura, Tunc
	Tunc = Tc*(1 + 0.3*(np.sin(theta)**2.2 + np.cos(tau/2)**3*(np.cos(eta)**2.2 - np.sin(theta)**2.2)))
	
	kp = getSWdata(epoch.date, f'KP{str(int(np.ceil(epoch.hour/3)))}')

	if altitude >= 200:
		deltaT_corr = 28*kp + 0.03*np.exp(kp)
	elif altitude < 200:
		deltaT_corr = 14*kp + 0.02*np.exp(kp)

	# Corrected exospheric temperature, Tcorr
	Tcorr = Tunc + deltaT_corr

	# Inflection point temperature, Tx
	Tx = 371.6678 + 0.0518806*Tcorr - 294.3505*np.exp(-0.00216222*Tcorr)

	# Base value temperature, T0, [K]
	T0 = 183

	if altitude <= 125:
		T_hellp = Tx + (Tx-T0)/35**4 * (-89284375*hellp**0 + 3542400*hellp**1 - 52687.5*hellp**2 + 340.5*hellp**3 - 0.8*hellp**4)
	elif altitude > 125:
		l = 0.1031445e5*Tcorr**0 + 0.234123e1*Tcorr**1 + 0.1579202e-2*Tcorr**2 - 0.1252487e-5*Tcorr**3 + 0.2462708e-9*Tcorr**4
		Rpole = Earth.radius*(1-Earth.reciprocal_flattening)

		T_hellp = Tcorr - (Tcorr-Tx) * np.exp((T0-Tx)/(Tcorr-Tx) * (hellp-125)/35 * (l/(Rpole+hellp)))

	return T_hellp


def dragAcceleration(r_ITRF, v_ITRF, CD, A, m):
	density = exponentialDensity(r_ITRF)
	v_rel = v_ITRF
	mod_v_rel = np.linalg.norm(v_rel)

	bc = (A * CD) / m

	a = -0.5 * 1000 * density * bc * mod_v_rel**2 * (v_rel/mod_v_rel)
	return a