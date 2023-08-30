"""1980 IAU Theory of Nutation Coefficients, Epoch J2000

The units for the longitude terms (Ai, Bi) and the obliquity terms (Ci, Di) are
0.0001" per Julian century."""

import pandas as pd
import numpy as np


nut_file = './data/iau80/nut80.csv'

conversion_factor = 0.0001 * np.pi / (180*3600)

nut80 = pd.read_csv(nut_file)
nut80.loc[:,"Ai":"Di"] *= conversion_factor