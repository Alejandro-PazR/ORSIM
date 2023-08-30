import datetime

import numpy as np

from data.EOP import getEOPdata
from transformations.angles import secToRad


def timeToHMS(time):
	temp = time/3600

	h = np.trunc(temp)
	min = np.trunc((temp - h)*60)
	s = np.around((temp - h - min/60)*3600, 16)

	return h, min, s


def HMStoTime(h, min, s):
	return np.around(3600*h + 60*min + s, 16)


def julianDate(yr, mo, d, h, min, s): # From 1900 to 2100
	date = datetime.datetime(yr, mo, d, int(h), int(min), int(s), 0)

	if getEOPdata(date, 'DAT', interpolate=True) % 1 == 0.0:
		leapSecond = 60
	else: # There is a leap second
		leapSecond = 61
	
	jd = np.around(367*yr
			- np.trunc(7*(yr+np.trunc((mo+9)/12))/4)
			+ np.trunc(275*mo/9)
			+ d
			+ 1721013.5
			+ (((s/leapSecond+min)/60)+h)/24, 16)
	
	return jd


def thetaGMST(T_UT1):
	"""Greenwich Mean Sidereal Time."""
	
	thetaGMST = 67310.54841 + (876600*3600 + 8640184.812866)*T_UT1 \
				+ 0.093104*T_UT1**2 - 6.2e-6*T_UT1**3
	
	thetaGMST = secToRad(thetaGMST*15) % (2*np.pi)

	if thetaGMST < 0:
		thetaGMST += 2*np.pi
	
	return thetaGMST