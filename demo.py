import numpy as np
import matplotlib.pyplot as plt

from core.spacecraft import SpaceCraft
from core.time import Epoch
from core.orbit import Orbit


# ----- EPOCH -----
epoch = Epoch (2023, 'Mar', 20, 21, 24, 15.386145)

# ----- SPACECRAFT -----
satellite = SpaceCraft(mass=850, CD=2.0, A_drag=15, CR=1.8, A_srp=15)

# ----- ELEMENTS -----
sma = 6932.1363
ecc = 0.004
inc = np.radians(23.6)
raan = np.radians(25.6)
aop = np.radians(98.5)
ta  = np.radians(12.2)

# ----- PROPAGATION SETTINGS -----
simulation_time = 86400 # In seconds

# ----- ORBIT PROPAGATION -----
orb1 = Orbit.from_elements(epoch, satellite, sma, ecc, inc, raan, aop, ta)
orb1.setPerturbations(
	gravity = False,
	drag = False,
	third_body = False,
	solar_radiation_pressure = False,
	albedo = False,
	longwave = False,
)
trajectory = orb1.propagate(simulation_time, method='Cowell')

# ----- RADIATION -----
solar,albedo,olr = orb1.radiation(gridSpacing=5)

# ----- TRAJECTORY PLOT -----
orb1.plotTrajectory(daynight=True)

plt.xlabel("Tiempo desde la época [s]")
plt.ylabel("Irradiación [W/m^-2]")
labels = ["Emisividad", "Solar", "Albedo"]

plt.stackplot(trajectory.t, olr, solar, albedo, labels=labels)
plt.legend(loc='upper left')
plt.show()