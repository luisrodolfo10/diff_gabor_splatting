"""Compile the CUDA extension against the active PyTorch installation."""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    ext_modules=[
        CUDAExtension(
            "gabor_splatting._C",
            [
                "src/gabor_splatting/csrc/bindings.cpp",
                "src/gabor_splatting/csrc/rasterize.cu",
                "src/gabor_splatting/csrc/spherical_harmonics.cu",
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-lineinfo"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
