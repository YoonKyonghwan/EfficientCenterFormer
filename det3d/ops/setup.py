"""
Please run at the project's root directory
python det3d/ops/setup.py develop
"""
import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_git_commit_number():
    if not os.path.exists('.git'):
        return '0000000'

    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext


if __name__ == '__main__':
    version = '0.1.0+%s' % get_git_commit_number()

    setup(
        name='ops',
        version=version,
        description='OpenPCDet is a general codebase for 3D object detection from point cloud',
        install_requires=[],

        author='Jeongwon Her',
        author_email='jwher96@snu.ac.kr',
        license='Apache License 2.0',
        packages=[],
        cmdclass={
            'build_ext': BuildExtension,
        },
        ext_modules=[
            make_cuda_ext(
                name='ingroup_inds_cuda',
                module='det3d.ops.ingroup_inds',
                sources=[
                    'src/ingroup_inds.cpp',
                    'src/ingroup_inds_kernel.cu',
                ]
            ),
        ],
    )
