from functools import cached_property
from datetime import datetime, timedelta
from calendar import month_abbr
from typing import Any

import numpy as np
from numpy import sin

from data.EOP import getEOPdata
from transformations.time import HMStoTime, julianDate, timeToHMS


class Epoch:
	def __init__(self, yr: int, mo: int | str, day: int, h: int, mins: int, s: float):
		if isinstance(mo, int):
			_mo = mo
		elif isinstance(mo, str):
			_mo = list(month_abbr).index(mo)
	
		self._s = s
		_ms = int(np.around(s - int(self._s),6)*1e6)
		self._date = datetime(yr, _mo, day, h, mins, int(self._s), _ms)

	def __str__(self) -> str:
		return f'Epoch: {self.date.strftime("%d-%B-%Y, %H:%M:%S")} UTC'

	def __repr__(self) -> str:
		return f'Epoch({self.date.year}, {self.date.month}, {self.date.day}, {self.date.hour}, {self.date.minute}, {self.s}, {self.ms})'

	@cached_property
	def date(self):
		return self._date

	@cached_property
	def UTC(self):
		"""Universal Time Coordinated."""
		return HMStoTime(self.date.hour, self.date.minute, self._s)
	
	@cached_property
	def JD_UTC(self):
		"""Julian date of UTC."""
		if self.UTC > 86400:
			new_date = self.date + timedelta(days=1)
			return julianDate(new_date.year, new_date.month, new_date.day, *timeToHMS(self.UTC%86400))
		else:
			return julianDate(self.date.year, self.date.month, self.date.day, *timeToHMS(self.UTC))

	@cached_property
	def UT1(self):
		"""UTC corrected for polar movement."""
		delta_UT1 = getEOPdata(self.date, 'UT1-UTC')
		return self.UTC + delta_UT1

	@cached_property
	def JD_UT1(self):
		"""Julian date of UT1."""
		if self.UT1 > 86400:
			new_date = self.date + timedelta(days=1)
			return julianDate(new_date.year, new_date.month, new_date.day, *timeToHMS(self.UT1%86400))
		else:
			return julianDate(self.date.year, self.date.month, self.date.day, *timeToHMS(self.UT1))

	@cached_property
	def T_UT1(self):
		"""Julian centuries of UT1."""
		return np.around((self.JD_UT1 - 2451545)/36525, 16)

	@cached_property
	def TAI(self):
		"""International Atomic Time."""
		delta_AT = getEOPdata(self.date, 'DAT', interpolate=False)
		return self.UTC + delta_AT

	@cached_property
	def TT(self):
		"""Terrestrial Time."""
		return self.TAI + 32.184

	@cached_property
	def JD_TT(self):
		"""Julian date of TT."""
		if self.TT > 86400:
			new_date = self.date + timedelta(days=1)
			return julianDate(new_date.year, new_date.month, new_date.day, *timeToHMS(self.TT%86400))
		else:
			return julianDate(self.date.year, self.date.month, self.date.day, *timeToHMS(self.TT))

	@cached_property
	def T_TT(self):
		"""Julian centuries of TT."""
		return np.around((self.JD_TT - 2451545)/36525, 16)

	@cached_property
	def TDB(self):
		"""Barycentric Dynamical Time."""
		return (self.TT + 0.001657*sin(628.3076*self.T_TT + 6.2401)
				+ 0.000022*sin(575.3385*self.T_TT + 4.2970) + 0.000014*sin(1256.6152*self.T_TT + 6.1969)
				+ 0.000005*sin(606.9777*self.T_TT + 4.0212) + 0.000005*sin(52.9691*self.T_TT + 0.4444)
				+ 0.000002*sin(21.3299*self.T_TT + 5.5431) + 0.000010*self.T_TT*sin(628.3076*self.T_TT + 4.2490))

	@cached_property
	def JD_TDB(self):
		"""Julian date of TDB."""
		if self.TDB > 86400:
			new_date = self.date + timedelta(days=1)
			return julianDate(new_date.year, new_date.month, new_date.day, *timeToHMS(self.TDB%86400))
		else:
			return julianDate(self.date.year, self.date.month, self.date.day, *timeToHMS(self.TDB))

	@cached_property
	def T_TDB(self):
		"""Julian centuries of TDB."""
		return np.around((self.JD_TDB - 2451545)/36525, 16)