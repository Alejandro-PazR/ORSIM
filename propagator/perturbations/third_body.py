from datetime import timedelta

import numpy as np
from scipy.interpolate import CubicSpline

from core.time import Epoch
from phenomena import Sun, Moon


def thirdBodyAcceleration(body, r_third, r_sat):
	r_sat3 = r_third - r_sat
	r_sat3_cb = np.linalg.norm(r_sat3)**3

	r_earth3 = r_third
	r_earth3_cb = np.linalg.norm(r_earth3)**3

	a = body.mu * (r_sat3/r_sat3_cb - r_earth3/r_earth3_cb)
	return a


def ephemerisInterpolation(body, epoch, tStop, points=20):
	times = np.linspace(0, tStop, num=points)
	vectors = np.empty((points, 3))
	
	for i, t in enumerate(times):
		new = epoch.date + timedelta(seconds=t)
		new_epoch = Epoch(new.year, new.month, new.day, new.hour, new.minute, new.second + new.microsecond*1e-6)

		if body == Sun:
			r_third = Sun(new_epoch).vector()
		elif body == Moon:
			r_third = Moon(new_epoch).vector()

		vectors[i] = r_third

	return CubicSpline(times, vectors)