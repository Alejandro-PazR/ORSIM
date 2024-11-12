import numpy as np
import matplotlib.pyplot as plt

from core.time import Epoch
from core.spacecraft import SpaceCraft
from core.orbit import Orbit


# ---------------------------------------- EPOCH --------------------------------------- #
epoch = Epoch(yr=2023, mo="Mar", day=20, h=21, mins=24, s=15.386145)

# ------------------------------------- SPACECRAFT ------------------------------------- #
satellite = SpaceCraft(mass=850, CD=2.0, A_drag=15, CR=1.8, A_srp=15)

# --------------------------------- Keplerian Elements --------------------------------- #
sma = 6932.1363  # [km]
ecc = 0.004
inc = np.radians(23.6)  # [rad]
raan = np.radians(25.6)  # [rad]
aop = np.radians(98.5)  # [rad]
ta = np.radians(12.2)  # [rad]

# -------------------------------- ORBIT INITIALISATION -------------------------------- #
orb1 = Orbit.from_elements(epoch, satellite, sma, ecc, inc, raan, aop, ta)
orb1.setPerturbations(
    gravity=False,
    drag=False,
    third_body=False,
    solar_radiation_pressure=False,
    albedo=False,
    longwave=False,
)

# ---------------------------------- ORBIT PROPAGATION --------------------------------- #
simulation_time = 86400  # In seconds
trajectory = orb1.propagate(simulation_time, method="Cowell")

# ----------------------------------- Trajectory plot ---------------------------------- #
orb1.plotTrajectory(daynight=True)

# -------------------------------------- RADIATION ------------------------------------- #
solar, albedo, olr = orb1.radiation(gridSpacing=5)

plt.xlabel("Time elapsed since epoch [s]")
plt.ylabel("Irradiation [W/m^-2]")
labels = ["Emissivity", "SRP", "Albedo"]

plt.stackplot(trajectory.t, olr, solar, albedo, labels=labels)
plt.legend(loc="upper left")
plt.show()
