from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_operator',
    ext_modules=[
        CUDAExtension(
            name='cuda_operator',
            sources=[
                'operator_cuda.cpp',
                'operator_cuda_kernel.cu',
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
