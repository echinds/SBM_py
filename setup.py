from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import os
import sys



os.environ["CC"]="/usr/local/bin/g++-10"
os.environ["CXX"]="/usr/local/bin/gcc-10"
ext_modules=[
    Extension("C_MonteCarlo",
              ["C_MonteCarlo.pyx"],
              libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math","-march=native", "-fopenmp","-std=c++11","-I/usr/local/include" ],
                  extra_link_args=['-lomp','-lgomp',"-fopenmp"],
              include_dirs = [np.get_include()]
              ) 
]
setup( 
  name = "C_MonteCarlo",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules
)