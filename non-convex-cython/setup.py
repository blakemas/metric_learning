from setuptools import setup, find_packages
try:
    from Cython.Build import cythonize
except ImportError:
    from pip import pip

    pip.main(['install', 'cython'])

    from Cython.Build import cythonize
import numpy as np

setup(
    name='utilsMetric',
    description='Ordinal Metric Learning with non-convex factored solver',
    ext_modules=cythonize(['utilsMetric.pyx']),
    include_dirs=[np.get_include()]
)
