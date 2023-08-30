import numpy as np

from data.EOP import getEOPdata
from data.iau80.nut80 import nut80

from transformations.rotations import R1, R2, R3
from transformations.angles import secToRad
from transformations.time import thetaGMST


class FK5:
	def __init__(self, epoch):
		self._epoch = epoch
		self.T_TT = epoch.T_TT
		self.T_UT1 = epoch.T_UT1

		self.xp = getEOPdata(epoch.date, 'X')
		self.yp = getEOPdata(epoch.date, 'Y')
		self.dd_psi = getEOPdata(epoch.date, 'DPSI')
		self.dd_eps = getEOPdata(epoch.date, 'DEPS')
		self.LOD = getEOPdata(epoch.date, 'LOD')

		self.P = FK5.precession(self.T_TT)
		self.N = self.nutation(self.T_TT, self.dd_psi, self.dd_eps)
		self.R = self.siderealTime(self.T_UT1, self.T_TT, self.delta_psi, self.mean_eps)
		self.W = self.polarMotion(self.xp, self.yp)

	def polarMotion(self, xp, yp):
		W = R1(secToRad(yp)) @ R2(secToRad(xp))
		return W

	def siderealTime(self, T_UT1, T_TT, delta_psi, mean_eps):
		_, _, _, _, Omega_moon = FK5.delaunay(T_TT)

		self.eq_equinox_star = delta_psi*np.cos(mean_eps)

		eq_equinox = self.eq_equinox_star \
			+ secToRad(0.00264)*np.sin(Omega_moon) \
			+ secToRad(0.000063)*np.sin(2*Omega_moon)
		
		theta_gast = thetaGMST(T_UT1) + eq_equinox
		
		R = R3(-theta_gast)
		return R

	@staticmethod
	def precession(T_TT):
		"""Converts mean equinox of date (MOD) to a vector in J2000."""

		# Combined effects of general precession
		zeta = 2306.2181*T_TT + 0.30188*T_TT**2 + 0.017998*T_TT**3
		theta = 2004.3109*T_TT - 0.42665*T_TT**2 - 0.041833*T_TT**3
		z = 2306.2181*T_TT + 1.09468*T_TT**2 + 0.018203*T_TT**3

		# Precession matrix
		P = R3(secToRad(zeta)) @ R2(secToRad(-theta)) @ R3(secToRad(z))
		return P

	def nutation(self, T_TT, dd_psi, dd_eps):
		M_moon, M_sun, u_M_moon, D_sun, Omega_moon = FK5.delaunay(T_TT)

		ap = nut80.loc[:, "an1"]*M_moon \
			+ nut80.loc[:, "an2"]*M_sun \
			+ nut80.loc[:, "an3"]*u_M_moon \
			+ nut80.loc[:, "an4"]*D_sun \
			+ nut80.loc[:, "an5"]*Omega_moon
		
		# Nutation in longitude
		delta_psi = ((nut80.loc[:, "Ai"] + nut80.loc[:, "Bi"]*T_TT)*np.sin(ap)).sum()
		
		# Nutation in obliquity
		self.mean_eps = np.radians(23.439291 - 0.0130042*T_TT - 1.64e-7*T_TT**2 + 5.04e-7*T_TT**3)
		delta_eps = ((nut80.loc[:, "Ci"] + nut80.loc[:, "Di"]*T_TT)*np.cos(ap)).sum()
		self.true_eps = self.mean_eps + delta_eps

		delta_psi = (delta_psi + secToRad(dd_psi))
		delta_eps = (delta_eps + secToRad(dd_eps))

		self.delta_psi = np.sign(delta_psi)*(np.abs(delta_psi) % (2*np.pi))
		delta_eps = np.sign(delta_eps)*(np.abs(delta_eps) % (2*np.pi))

		# Nutation matrix
		N = R1(-self.mean_eps) @ R3(self.delta_psi) @ R1(self.true_eps)
		return N
	
	@property
	def epoch(self):
		return self._epoch
	
	@staticmethod
	def earthRotationalVelocity(LOD):
		omega_earth = np.around(7.292115146706979e-5 * (1 - LOD/86400), 16)
		return np.array([0, 0, omega_earth])

	@staticmethod
	def delaunay(T_TT):
		"""
		Reference
		---------
		Dennis D. McCarthy, IERS Technical Note No. 13 page 32
		"""
		
		# Mean Anomaly of the Moon
		M_moon = ((((0.064) * T_TT + 31.310) * T_TT + 1717915922.6330) * T_TT) / 3600.0 + 134.96298139
		# Mean Anomaly of the Sun	
		M_sun = ((((-0.012) * T_TT - 0.577) * T_TT + 129596581.2240) * T_TT) / 3600.0 + 357.52772333
		# Mean Argument of latitude of the Moon measured on the ecliptic from the mean equinox of date
		u_M_moon = ((((0.011) * T_TT - 13.257) * T_TT + 1739527263.1370) * T_TT) / 3600.0 + 93.27191028
		# Mean Elongation from the Sun
		D_sun = ((((0.019) * T_TT - 6.891) * T_TT + 1602961601.3280) * T_TT) / 3600.0 + 297.85036306
		# Mean Longitude of the Ascending Node of the Moon
		Omega_moon = ((((0.008) * T_TT + 7.455) * T_TT - 6962890.5390) * T_TT) / 3600.0 + 125.04452222

		values = [M_moon, M_sun, u_M_moon, D_sun, Omega_moon]

		return [np.radians(np.mod(a,360)) for a in values]

	@classmethod
	def TEME_to_J200(cls, epoch, r_TEME):
		fk5 = cls(epoch)
		return fk5.P @ fk5.N @ R3(-fk5.eq_equinox_star) @ r_TEME
	
	@staticmethod
	def MOD_to_J2000(T_TT, r_MOD):
		return FK5.precession(T_TT) @ r_MOD
	
	def GCRF_to_ITRF(self, r_GCRF, v_GCRF=np.zeros(3), a_GCRF=np.zeros(3)):
		r_MOD = self.P.T @ r_GCRF
		v_MOD = self.P.T @ v_GCRF
		a_MOD = self.P.T @ a_GCRF

		r_TOD = self.N.T @ r_MOD
		v_TOD = self.N.T @ v_MOD
		a_TOD = self.N.T @ a_MOD

		r_PEF = self.R.T @ r_TOD
		v_PEF = self.R.T @ v_TOD
		a_PEF = self.R.T @ a_TOD

		omega_vec = self.earthRotationalVelocity(self.LOD)

		r_ITRF = self.W.T @ r_PEF
		v_ITRF = self.W.T @ (v_PEF - np.cross(omega_vec, r_PEF))
		a_ITRF = self.W.T @ (a_PEF - np.cross(omega_vec, np.cross(omega_vec, r_PEF)) - 2*np.cross(omega_vec, v_PEF))

		return r_ITRF, v_ITRF, a_ITRF

	def ITRF_to_GCRF(self, r_ITRF, v_ITRF=np.zeros(3), a_ITRF=np.zeros(3)):
		r_PEF = self.W @ r_ITRF
		v_PEF = self.W @ v_ITRF
		a_PEF = self.W @ a_ITRF

		omega_vec = FK5.earthRotationalVelocity(self.LOD)

		r_TOD = self.R @ r_PEF
		v_TOD = self.R @ (v_PEF + np.cross(omega_vec, r_PEF))
		a_TOD = self.R @ (a_PEF + np.cross(omega_vec, np.cross(omega_vec, r_PEF)) + 2*np.cross(omega_vec, v_PEF))

		r_MOD = self.N @ r_TOD
		v_MOD = self.N @ v_TOD
		a_MOD = self.N @ a_TOD

		r_GCRF = self.P @ r_MOD
		v_GCRF = self.P @ v_MOD
		a_GCRF = self.P @ a_MOD

		return r_GCRF, v_GCRF, a_GCRF