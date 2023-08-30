import numpy as np
from numpy import sin, cos
import pandas as pd
import io

from data.EOP import getEOPdata

from transformations.rotations import R1, R2, R3
from transformations.angles import *


class IAU2006:
	def __init__(self, epoch):
		self._epoch = epoch
		self.T_TT = epoch.T_TT
		self.T_TDB = epoch.T_TDB
		self.JD_UT1 = epoch.JD_UT1

		self.xp = getEOPdata(epoch.date, 'X')
		self.yp = getEOPdata(epoch.date, 'Y')
		self.dX = getEOPdata(epoch.date, 'DX')
		self.dY = getEOPdata(epoch.date, 'DY')
		self.LOD = getEOPdata(epoch.date, 'LOD')

		self.PN = self.precessionNutation(self.dX, self.dY, self.T_TT, self.T_TDB)
		self.R = self.earthRotationAngle(self.JD_UT1)
		self.W = self.polarMotion(self.T_TT, self.xp, self.yp)
	
	def polarMotion(self, T_TT, xp, yp):
		"""Returns polar motion transformation matrix.

		Notes
		-----
		Vallado:Polar Motion:p.212

		The transformation matrix arising from polar motion	(i.e. relating ITRS	and TIRS)
		can be expressed as:

		.. math:: W(t) = R_3(-s') \cdot R_2(x_p) \cdot R_1(y_p),

		:math:`xp` and :math:`yp` being the "polar coordinates" of the Celestial Intermediate Pole (CIP)
		in the ITRS and s' being a quantity, named "TIO locator", which provides the
		position of the TIO on the equator of the CIP corresponding to the kinematical
		definition of the "non-rotating" origin (NRO) in the ITRS when the CIP is moving
		with respect to the ITRS due to polar motion.
		The main component of :math:`s'` is found using the average values for the Chandler wobble (a_e = 0.26")
		and the annual wobble (a_a = 0.12") of the pole. It is less than 0.0004" for the next century.

		Parameters
		----------
		xp : float, [arcseconds]
			Polar coordinate of the Celestial Intermediate Pole (CIP).
		yp : float, [arcseconds]
			Polar coordinate of the Celestial Intermediate Pole (CIP).
		T_TT : float
			Julian centuries of terrestrial time.

		Returns
		-------
		W : np.array
			3x3 rotation matrix  from ITRF to TIRS coordinate system.
		"""

		s_prime = secToRad(-0.000047/3600)*T_TT	
		return R3(-s_prime) @ R2(secToRad(xp)) @ R1(secToRad(yp))

	def earthRotationAngle(self, JD_UT1):
		theta_ERA = (2*np.pi*(0.7790572732640 + 1.00273781191135448*(JD_UT1 - 2451545.0)))
		return R3(-theta_ERA)

	def precessionNutation(self, dX, dY, T_TT, T_TDB):

		X = self.mainCoefficient('X', T_TT, T_TDB) + secToRad(dX)
		Y = self.mainCoefficient('Y', T_TT, T_TDB) + secToRad(dY)
		s = self.mainCoefficient('s', T_TT, T_TDB) - (X*Y)/2

		a = 1/(1+cos(np.arctan(np.sqrt((X**2 + Y**2)/(1-X**2-Y**2)))))

		return np.array([[1-a*X**2, -a*X*Y, X],
					   [-a*X*Y, 1-a*Y**2, Y],
					   [-X, -Y, 1-a*(X**2+Y**2)]]) @ R3(s)

	def mainCoefficient(self, coefficient: str, T_TT, T_TDB):
		file_path = './data/iau2006/'
		coef_files = {'X': 'tab5.2a.txt', 'Y': 'tab5.2b.txt', 's': 'tab5.2d.txt'}
		poly_coefs = {'X': [-0.016617, 2004.191898, -0.4297829, -0.19861834, 0.000007578, 0.0000059285],
					  'Y': [-0.006951, -0.025896, -22.4072747, 0.00190059, 0.001112526, 0.0000001358],
					  's': [0.000094, 0.00380865, -0.00012268, -0.07257411, 0.00002798, 0.00001562]}
		
		def nutationCoefficientsTable(file, j):
			s = open(file, 'r').read()
			# Remove all text at the beginning
			text_no_header = s[s.find('j = 0'):]
			lines = text_no_header.split('\n')  # Split in list

			j_values = [i for i in lines if i.find('j =') is not -1]
			j_values = [0] + [int(a.split()[-1]) for a in j_values]
			j_values = np.cumsum(j_values)

			lines = [i for i in lines if not i == '' or i == ' ']  # Remove all blank lines
			# Remove lines with text
			lines = [i for i in lines if i.find('j =') is -1]
			s = '\n'.join([";".join(a.split()) for a in lines])  # Convert into csv format

			col_names = ["i", "as", "ac", "l", "l'", "F", "D", "Om", "L_Me",
						 "L_Ve", "L_E", "L_Ma", "L_J", "L_Sa", "L_U", "L_Ne", "p_A"]
			df = pd.read_table(io.StringIO(s), sep=";", header=None, names=col_names, index_col=0)

			return df.loc[j_values[j]+1:j_values[j+1],]
		
		polynomial_part = 0
		for j in range(0, 6):
			polynomial_part += secToRad(poly_coefs[coefficient][j] * T_TT**j)

		non_polynomial_part = 0
		for j in range(0, 5):
			nutation_coefs = nutationCoefficientsTable(file_path + coef_files[coefficient], j)

			a_s = secToRad(nutation_coefs.loc[:, "as"] / 1e6)
			a_c = secToRad(nutation_coefs.loc[:, "ac"] / 1e6)

			arg = pd.concat([nutation_coefs.loc[:, "l":"Om"] * IAU2006.luniSolarNutationAngles(T_TDB),
								nutation_coefs.loc[:, "L_Me":"p_A"] * IAU2006.planetaryNutationAngles(T_TDB)],
							axis=1).sum(1)

			non_polynomial_part += ((a_s*np.sin(arg) + a_c*np.cos(arg))*T_TT**j).sum()

		return (polynomial_part + non_polynomial_part)

	@staticmethod
	def luniSolarNutationAngles(T_TT):
		M_moon = 485868.249036 + 1717915923.2178*T_TT + 31.8792*T_TT**2 + 0.051635*T_TT**3 - 0.00024470*T_TT**4
		M_sun = 1287104.79305 + 129596581.0481*T_TT - 0.5532*T_TT**2 + 0.000136*T_TT**3 - 0.00001149*T_TT**4
		u_M_moon = 335779.526232 + 1739527262.8478*T_TT - 12.7512*T_TT**2 - 0.001037*T_TT**3 + 0.00000417*T_TT**4
		D_sun = 1072260.70369 + 1602961601.2090*T_TT - 6.3706*T_TT**2 + 0.006593*T_TT**3 - 0.00003169*T_TT**4
		Omega_moon = 450160.398036 - 6962890.5431*T_TT + 7.4722*T_TT**2 + 0.007702*T_TT**3 - 0.00005939*T_TT**4

		values = [M_moon, M_sun, u_M_moon, D_sun, Omega_moon]
		return [np.radians(np.mod(a,360)) for a in values]

	@staticmethod
	def planetaryNutationAngles(T_TDB):
		lambda_M_mercury = 252.250905494 + 149472.6746358*T_TDB
		lambda_M_venus = 181.979800853 + 58517.8156748*T_TDB
		lambda_M_earth = 100.466448494 + 35999.3728521*T_TDB
		lambda_M_mars = 355.433274605 + 19140.299314*T_TDB
		lambda_M_jupiter = 34.351483900 + 3034.90567464*T_TDB
		lambda_M_saturn = 50.0774713998 + 1222.11379404*T_TDB
		lambda_M_uranus = 314.055005137 + 428.466998313*T_TDB
		lambda_M_neptune = 304.348665499 + 218.486200208*T_TDB
		p_lambda = 1.39697137214*T_TDB + 0.0003086*T_TDB**2

		values = [lambda_M_mercury, lambda_M_venus, lambda_M_earth, lambda_M_mars, lambda_M_jupiter,
				  lambda_M_saturn, lambda_M_uranus, lambda_M_neptune, p_lambda]
		return [np.radians(np.mod(a,360)) for a in values]

	@staticmethod
	def earthRotationalVelocity(LOD):
		omega_earth = np.around(7.292115146706979e-5 * (1 - LOD/86400), 16)
		return np.array([0, 0, omega_earth])
	
	@property
	def epoch(self):
		return self._epoch
	
	@classmethod
	def ITRF_to_GCRF(cls, r_ITRF, epoch):
		iau2006 = cls(epoch)

		r_TIRS = np.squeeze(np.asarray(np.matmul(iau2006.W, r_ITRF)))
		r_CIRS = np.squeeze(np.asarray(np.matmul(iau2006.R, r_TIRS)))
		r_GCRF = np.squeeze(np.asarray(np.matmul(iau2006.PN, r_CIRS)))

		print(f"r_ITRF: {r_ITRF}")
		print(f"r_TIRS: {r_TIRS}")
		print(f"r_CIRS: {r_CIRS}")
		print(f"r_GCRF: {r_GCRF}")