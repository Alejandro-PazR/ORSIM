import numpy as np

from constants.general import Earth


def J2_perturbation(J2, mu, radius, r, r_I, r_J, r_K):
	a_I = ((-3*J2*r_I*mu*radius**2)/(2*r**5)) * (1 - (5*r_K**2)/(r**2))
	a_J = ((-3*J2*r_J*mu*radius**2)/(2*r**5)) * (1 - (5*r_K**2)/(r**2))
	a_K = ((-3*J2*r_K*mu*radius**2)/(2*r**5)) * (3 - (5*r_K**2)/(r**2))
	return np.array([a_I, a_J, a_K])


def J3_perturbation(J3, mu, radius, r, r_I, r_J, r_K):
	a_I = (-5*J3*mu*radius**3*r_I)/(2*r**7) * (3*r_K - (7*r_K**3)/r**2)
	a_J = (-5*J3*mu*radius**3*r_J)/(2*r**7) * (3*r_K - (7*r_K**3)/r**2)
	a_K = (-5*J3*mu*radius**3)/(2*r**7) * (6*r_K**2 - (7*r_K**4)/r**2 - (3*r**2)/5)

	return np.array([a_I, a_J, a_K])


def J4_perturbation(J4, mu, radius, r, r_I, r_J, r_K):
	a_I = ((15*J4*mu*radius**4*r_I)/(8*r**7)) * (1 - (14*r_K**2)/(r**2) + (21*r_K**4)/(r**4))
	a_J = ((15*J4*mu*radius**4*r_J)/(8*r**7)) * (1 - (14*r_K**2)/(r**2) + (21*r_K**4)/(r**4))
	a_K = ((15*J4*mu*radius**4*r_K)/(8*r**7)) * (5 - (70*r_K**2)/(3*r**2) + (21*r_K**4)/(r**4))

	return np.array([a_I, a_J, a_K])


def J5_perturbation(J5, mu, radius, r, r_I, r_J, r_K):
	a_I = (3*J5*mu*radius**5*r_I*r_K)/(8*r**9) * (35 - (210*r_K**2)/r**2 + (231*r_K**4)/r**4)
	a_J = (3*J5*mu*radius**5*r_J*r_K)/(8*r**9) * (35 - (210*r_K**2)/r**2 + (231*r_K**4)/r**4)
	a_K = (3*J5*mu*radius**5*r_K*r_K)/(8*r**9) * (105 - (315*r_K**2)/r**2 + (231*r_K**4)/r**4) - (15*J5*mu*radius**5)/(8*r**7)

	return np.array([a_I, a_J, a_K])


def J6_perturbation(J6, mu, radius, r, r_I, r_J, r_K):
	a_I = (-J6*mu*radius**6*r_I)/(16*r**9) * (35 - (945*r_K**2)/r**2 + (3465*r_K**4)/r**4 - (3003*r_K**6)/r**6)
	a_J = (-J6*mu*radius**6*r_J)/(16*r**9) * (35 - (945*r_K**2)/r**2 + (3465*r_K**4)/r**4 - (3003*r_K**6)/r**6)
	a_K = (-J6*mu*radius**6*r_K)/(16*r**9) * (245 - (2205*r_K**2)/r**2 + (4851*r_K**4)/r**4 - (3003*r_K**6)/r**6)

	return np.array([a_I, a_J, a_K])

def asphericalAcceleration(r_ITRF):
	r_I, r_J, r_K = r_ITRF.T
	r = np.linalg.norm(r_ITRF)

	args = (Earth.mu, Earth.radius, r, r_I, r_J, r_K)
	
	a = J2_perturbation(Earth.J2, *args) + J3_perturbation(Earth.J3, *args) + J4_perturbation(Earth.J4, *args) + J5_perturbation(Earth.J5, *args) + J6_perturbation(Earth.J6, *args)
	return a


"""
import numpy as np
import pandas as pd
from scipy.special import legendre

from constants.general import Earth
from transformations.coordinates import ECEF_to_latlon


# https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84
gravitational_coeffs = pd.read_csv('./data/gravity/EGM08.csv')

def getCoef(coef_type, l, m):
	temp = gravitational_coeffs[gravitational_coeffs['l'] == l]
	coef = temp.loc[temp['m'] == m, coef_type].item()
	return coef


def legendreFunction(x, l, m):
	if m == 0: k = 1
	else: k = 2
	pilm = np.sqrt(np.math.factorial(l+m) / (np.math.factorial(l-m) * k * (2*l+1)))
	return (1 - x**2)**(m/2)*np.polyval(np.polyder(legendre(l), m), x) / pilm


def asphericalPotentialDeriv(r, phi_gc, lambda_sat):
	dU_dr = 0
	dU_dphi = 0
	dU_dlambda = 0

	temp = Earth.radius/r

	l = 2
	m = 0

	Clm = getCoef('Clm', l, m)
	Slm = getCoef('Clm', l, m)

	Plm_sin = legendreFunction(np.sin(phi_gc), l, m)
	Plmm_sin = legendreFunction(np.sin(phi_gc), l, m)

	dU_dr += temp**l * (l+1) * Plm_sin \
		* (Clm*np.cos(m*lambda_sat) + Slm*np.sin(m*lambda_sat))

	dU_dphi += temp**l * (Plmm_sin - m*np.tan(phi_gc)*Plm_sin) \
		* (Clm*np.cos(m*lambda_sat) + Slm*np.sin(m*lambda_sat))

	dU_dlambda += temp**l * m * Plm_sin \
		* (Slm*np.cos(m*lambda_sat) - Clm*np.sin(m*lambda_sat))
	
	dU_dr *= -(Earth.mu/r**2)
	dU_dphi *= (Earth.mu/r)
	dU_dlambda *= (Earth.mu/r)
	
	return dU_dr, dU_dphi, dU_dlambda


def asphericalAcceleration(r_ITRF):
	r = np.linalg.norm(r_ITRF)
	rI = r_ITRF[0]
	rJ = r_ITRF[1]
	rK = r_ITRF[2]

	phi_gc, lambda_sat, _ = ECEF_to_latlon(r_ITRF)

	dU_dr, dU_dphi, dU_dlambda = asphericalPotentialDeriv(r, phi_gc, lambda_sat)

	temp1 = (dU_dr/r) - (rK*dU_dphi)/(r**2*np.sqrt(rI**2 + rJ**2))
	temp2 = dU_dlambda/(rI**2 + rJ**2)

	aI = temp1*rI - temp2*rJ - Earth.mu*r/r**3
	aJ = temp1*rJ + temp2*rI - Earth.mu*r/r**3
	aK = (dU_dr * rK)/r + (np.sqrt(rI**2 + rJ**2)*dU_dphi)/r**2 - Earth.mu*r/r**3

	a_ITRF = np.array([aI, aJ, aK])
	return a_ITRF
"""