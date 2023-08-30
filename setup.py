from setuptools import setup

setup(
    name='astro',
    version='1.0',
    description='Orbital propagation.',
    author='Alejandro Paz Rodríguez',
    author_email='apazro00@estudiantes.unileon.es',
    packages=['astro'],
    install_requires=['numpy', 'pyvista', 'matplotlib', 'numba', 'scipy']
)