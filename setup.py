from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np


setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension('my_cython_code', ['fill.pyx'])],
    include_dirs=[np.get_include()])
