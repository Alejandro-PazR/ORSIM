import time
from constants.general import Earth

from propagator.perturbations.non_sphericity import J4_perturbation
import numpy as np

C2_adim = -0.000484165143790815
C3_adim = 9.57161207093473e-07
C4_adim = 5.39965866638991e-07
C5_adim = 6.86702913736681e-08
C6_adim = -1.49953927978527e-07

coefs = np.array([C2_adim, C3_adim, C4_adim, C5_adim, C6_adim])
pilm = np.array([])


for l in range(2, 7):
	pilm = np.append(pilm, np.sqrt(1/(2*l + 1)))

J = -coefs/pilm

for i, j in enumerate(J):
	print(f'J{i+2}: {j}')