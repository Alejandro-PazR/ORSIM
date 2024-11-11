import time
import pyvista as pv
import numpy as np
from numba import jit
from PIL import Image

from core.time import Epoch
from _math.ellipsoid import earthOrthodromic
from transformations.angles import eulerAngles
from constants.general import Earth
from phenomena import Sun
from reduction.fk5 import FK5


planet = pv.Sphere(
    radius=1, theta_resolution=240, phi_resolution=240, start_theta=270.001, end_theta=270
)

planet.active_texture_coordinates = np.zeros((planet.points.shape[0], 2))

planet.active_texture_coordinates[:, 0] = 0.5 + np.arctan2(
    -planet.points[:, 0], planet.points[:, 1]
) / (2 * np.pi)
planet.active_texture_coordinates[:, 1] = 0.5 + np.arcsin(planet.points[:, 2]) / np.pi

planet = planet.scale(Earth.radius)
planet = planet.rotate_z(87)


def orbitPlot(epoch, points, daynight=False):
    plotter = pv.Plotter()

    # ----- SCENE PROPERTIES -----
    plotter.add_text(
        str(epoch),
        font="courier",
        font_size=12,
        position="lower_right",
        color="white",
    )
    plotter.set_background(color=[0, 4, 10])
    plotter.add_camera_orientation_widget()
    plotter.show_axes()

    if daynight:
        earthTexture(epoch, gradient=0.002)
        tex = pv.read_texture("./plotter/day_night.png")
        light_intensity = 0.9
    else:
        tex = pv.read_texture("./plotter/bluemarble.jpg")
        light_intensity = 0.5

    # ----- LIGHTS -----
    for i in plotter.renderer.lights:
        if i.light_type != "Headlight":
            i.intensity *= light_intensity

    decl = np.degrees(Sun(epoch).declination)
    ra = np.degrees(Sun(epoch).rightAscension)

    sun = pv.Light(intensity=0.9)
    sun.set_direction_angle(elev=decl, azim=ra)
    plotter.add_light(sun)

    # ----- EARTH -----
    # Rotate the Earth from ECEF to ECI (J2000)

    i = np.array([1, 0, 0])
    j = np.array([0, 1, 0])
    k = np.array([0, 0, 1])

    reduction = FK5(epoch)
    i_J2000, _, _ = reduction.ITRF_to_GCRF(r_ITRF=i)
    j_J2000, _, _ = reduction.ITRF_to_GCRF(r_ITRF=j)
    k_J2000, _, _ = reduction.ITRF_to_GCRF(r_ITRF=k)

    matrix_J2000 = np.column_stack((i_J2000, j_J2000, k_J2000))
    alpha, beta, gamma = np.degrees(eulerAngles(matrix_J2000))

    earth_mesh = planet.rotate_z(gamma)
    earth_mesh = earth_mesh.rotate_x(beta)
    earth_mesh = earth_mesh.rotate_z(-alpha)

    earth = plotter.add_mesh(earth_mesh, smooth_shading=True, texture=tex, ambient=0)

    # ----- SATELLITE -----
    satellite = pv.Sphere(
        radius=300,
        center=points[-1],
        theta_resolution=120,
        phi_resolution=120,
        start_theta=270.001,
        end_theta=270,
    )
    plotter.add_mesh(satellite, color="g")

    # ----- ORBIT -----
    spline = pv.Spline(points, len(points) * 6)
    plotter.add_mesh(spline, line_width=1, color="r")

    plotter.show(title="OrbitPlotter")

    # for i in range(1, len(points)):
    # 	spline = pv.Spline(points[:i], len(points)*6)
    # 	plotter.add_mesh(spline, line_width=3, color='r')
    # 	plotter.update(10)


@jit
def getLon(x):
    return (x - 1024) * (360 / 2048)


@jit
def getLat(y):
    return (y - 512) * (180 / 1024)


@jit
def getX(lon):
    return round(lon * 2048 / 360) + 1024


@jit
def getY(lat):
    return round(lat * 1024 / 180) + 512


@jit
def sigmoid(x, gradient):
    distance = 20037.5
    alpha = 255 / (1 + np.exp(-gradient * (x - distance / 2)))
    if alpha < 0.3:
        alpha = 0
    elif alpha > 254.8:
        alpha = 255
    return alpha


def earthTexture(epoch, gradient=0.005):
    solar = Sun(epoch)

    lat_ss, lon_ss = solar.subsolar

    day = Image.open("./plotter/bluemarble.jpg").convert("RGBA")
    night = Image.open("./plotter/bluemarble_night.jpg").convert("RGBA")

    width, height = night.size

    npDay = np.array(day)
    npNight = np.array(night)

    for x in range(width):
        for y in range(height):
            pixel_lat = getLat(y)
            pixel_lon = getLon(x)

            distance = earthOrthodromic(-lat_ss, pixel_lat, lon_ss, pixel_lon)
            npDay[y, x, -1] = 255 - sigmoid(distance, gradient)
            npNight[y, x, -1] = sigmoid(distance, gradient)

    dayTransparent = Image.fromarray(npDay, "RGBA")
    nightTransparent = Image.fromarray(npNight, "RGBA")

    blended = Image.alpha_composite(dayTransparent, nightTransparent)
    blended.putalpha(255)
    blended.save("./plotter/day_night.png")
