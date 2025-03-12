from setuptools import setup
import os
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = os.path.dirname(__file__)

include_dirs = [
    os.path.join(ROOT_DIR, 'include'),
    os.path.join(ROOT_DIR, 'third_party', 'eigen'),
    os.path.join(ROOT_DIR, 'third_party', 'pybind11', 'include'),
    '/usr/local/include',
    '/usr/include',
    '/usr/local/cuda-11.1/targets/x86_64-linux/include'
]

sources = [
    'ifps-api.cu', 'src/ifps.cu'
]

macros = [
    ('EIGEN_GPUCC', None)
]

print(include_dirs)

module = CUDAExtension(
    name='PyIfps',
    sources=sources,
    include_dirs=include_dirs,
    define_macros=macros,
    extra_compile_args={
        'nvcc': [
                '--extended-lambda', 
                 "-Xcompiler=-mf16c",
                 "-Xcompiler=-Wno-float-conversion",
                 "-Xcompiler=-fno-strict-aliasing",
                 "-Xcompiler=-fPIC",
                 "--expt-relaxed-constexpr"],
        'cxx': ["--std=14"]
    }
)

setup(name='PyIfps',
      version='1.0',
      author='KKK',
      ext_modules=[module],
      cmdclass={
          'build_ext': BuildExtension
      })