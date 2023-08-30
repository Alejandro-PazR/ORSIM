"""
Space Weather Data
CelesTrak.org	Dr. T.S Kelso

DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,F10.7_OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_LAST81
2000-01-01,2272,7,53,47,40,33,43,30,43,37,327,56,39,27,18,32,15,32,22,30,1.3,6,71,129.9,125.6,OBS,166.2,179.0,161.1,175.0

Field		Description
-----		-----------
DATE		Year-Month-Day (ISO 8601)
BSRN		Bartels Solar Rotation Number. A sequence of 27-day intervals counted continuously from 1832 Feb 8.
ND			Number of Day within the Bartels 27-day cycle (01-27).
KP1			Planetary 3-hour Range Index (Kp) for 0000-0300 UT.
KP2			Planetary 3-hour Range Index (Kp) for 0300-0600 UT.
KP3			Planetary 3-hour Range Index (Kp) for 0600-0900 UT.
KP4			Planetary 3-hour Range Index (Kp) for 0900-1200 UT.
KP5			Planetary 3-hour Range Index (Kp) for 1200-1500 UT.
KP6			Planetary 3-hour Range Index (Kp) for 1500-1800 UT.
KP7			Planetary 3-hour Range Index (Kp) for 1800-2100 UT.
KP8			Planetary 3-hour Range Index (Kp) for 2100-0000 UT.
KP_SUM		Sum of the 8 Kp indices for the day.
			Kp has values of 0o, 0+, 1-, 1o, 1+, 2-, 2o, 2+, ... , 8o, 8+, 9-, 9o, which are expressed in steps 
				of one third unit. These values are multiplied by 10 and rounded to an integer value.
AP1			Planetary Equivalent Amplitude (Ap) for 0000-0300 UT.
AP2			Planetary Equivalent Amplitude (Ap) for 0300-0600 UT.
AP3			Planetary Equivalent Amplitude (Ap) for 0600-0900 UT.
AP4			Planetary Equivalent Amplitude (Ap) for 0900-1200 UT.
AP5			Planetary Equivalent Amplitude (Ap) for 1200-1500 UT.
AP6			Planetary Equivalent Amplitude (Ap) for 1500-1800 UT.
AP7			Planetary Equivalent Amplitude (Ap) for 1800-2100 UT.
AP8			Planetary Equivalent Amplitude (Ap) for 2100-0000 UT.
AP_AVG		Arithmetic average of the 8 Ap indices for the day.
CP			Cp or Planetary Daily Character Figure. A qualitative estimate of overall level of magnetic activity for 
				the day determined from the sum of the 8 Ap indices. Cp ranges, in steps of one-tenth, from 0 (quiet) 
				to 2.5 (highly disturbed).
C9			C9. A conversion of the 0-to-2.5 range of the Cp index to one digit between 0 and 9.
ISN			International Sunspot Number. Records contain the Zurich number through 1980 Dec 31 and 
				the International Brussels number thereafter.
F10.7_OBS			Observed 10.7-cm Solar Radio Flux (F10.7). Measured at Ottawa at 1700 UT daily from 
						1947 Feb 14 until 1991 May 31 and measured at Penticton at 2000 UT from 1991 Jun 01 on.
						Expressed in units of 10-22 W/m2/Hz.
F10.7_ADJ			10.7-cm Solar Radio Flux (F10.7) adjusted to 1 AU.
F10.7_DATA_TYPE		Flux Qualifier.
						OBS: Observed flux measurement
						INT: CelesTrak linear interpolation of missing data
						PRD: 45-Day predicted flux
						PRM: Monthly predicted flux
F10.7_OBS_CENTER81	Centered 81-day arithmetic average of F10.7 (observed).
F10.7_OBS_LAST81	Last 81-day arithmetic average of F10.7 (observed).
F10.7_ADJ_CENTER81	Centered 81-day arithmetic average of F10.7 (adjusted).
F10.7_ADJ_LAST81	Last 81-day arithmetic average of F10.7 (adjusted).
"""


import os
import datetime
import logging
from utils.log_format import CustomFormatter
from urllib import request

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


# Logging settings
logger = logging.getLogger("Space Weather")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)


url = 'http://celestrak.org/SpaceData/SW-All.csv'
SW_file = './data/SW/SW-All.csv'

m_time = os.path.getmtime(SW_file)
dt_m = datetime.datetime.fromtimestamp(m_time)
now = datetime.datetime.now()

file_time = now - dt_m
interval = datetime.timedelta(hours=9)

# If the file was downloaded more than 9h ago we try to update it.
if file_time > interval:
	try:
		request.urlretrieve(url, SW_file)
	except:
		logger.warning("Could not download SW file. Using not up-to-date local version.")
	else:
		logger.info("SW successfully downloaded.")
else:
	pass


SW = pd.read_csv(SW_file, index_col=0)


def getSWdata(date: datetime.datetime, field: str, interpolate: bool=True):
	current_date = date.strftime('%Y-%m-%d')
	next_date = (date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

	a = SW.loc[current_date, field]
	b = SW.loc[next_date, field]

	if interpolate:
		x = np.around(3600*date.hour + 60*date.minute + date.second + date.microsecond*1e-6, 16)
		interp_func = CubicSpline([0, 86400], [a, b])
		return float(interp_func(x))
	else:
		return a