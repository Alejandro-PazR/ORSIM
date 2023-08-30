"""
Earth Orientation Parameters
CelesTrak.org	Dr. T.S Kelso

DATE,MJD,X,Y,UT1-UTC,LOD,DPSI,DEPS,DX,DY,DAT,DATA_TYPE
2000-01-01,51544,0.043282,0.377909,0.3553880,0.0009536,-0.050263,-0.002471,-0.000005,-0.000050,32,O

Field		Description
-----		-----------
DATE		Year-Month-Day (ISO 8601)
MJD			Modified Julian Date (Julian Date at 0h UT minus 2400000.5)
X			x (arc seconds)
Y			y (arc seconds)
UT1-UTC		UT1-UTC (seconds)
LOD			Length of Day (seconds)
DPSI		δΔψ (arc seconds)
DEPS		δΔε (arc seconds)
DX			δX (arc seconds)
DY			δY (arc seconds)
DAT			Delta Atomic Time, TAI-UTC (seconds)
DATA_TYPE	O = Observed, P = Predicted
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
logger = logging.getLogger("Earth Orientation Parameters")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)


url = 'http://celestrak.org/SpaceData/EOP-All.csv'
EOP_file = './data/EOP/EOP-All.csv'

m_time = os.path.getmtime(EOP_file)
dt_m = datetime.datetime.fromtimestamp(m_time)
now = datetime.datetime.now()

file_time = now - dt_m
interval = datetime.timedelta(hours=9)

# If the file was downloaded more than 9h ago we try to update it.
if file_time > interval: 
	try:
		request.urlretrieve(url, EOP_file)
	except:
		logger.warning("Could not download EOP file. Using not up-to-date local version.")
	else:
		logger.info("EOP successfully downloaded.")
else:
	pass

EOP = pd.read_csv(EOP_file, index_col=0)

def getEOPdata(date: datetime.datetime, field: str, interpolate: bool=True):
	current_date = date.strftime('%Y-%m-%d')
	next_date = (date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

	a =	EOP.loc[current_date, field]
	b = EOP.loc[next_date, field]

	if interpolate:
		x = np.around(3600*date.hour + 60*date.minute + date.second + date.microsecond*1e-6, 16)
		interp_func = CubicSpline([0, 86400], [a, b])
		return float(interp_func(x))
	else:
		return a