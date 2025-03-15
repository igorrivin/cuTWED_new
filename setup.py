#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import platform
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_call(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection & inclusion of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        # Check if CUDA is available
        cuda_available = False
        try:
            if os.environ.get('FORCE_CUDA', '0') == '1':
                cuda_available = True
                print("Forcing CUDA build as requested by FORCE_CUDA environment variable")
            elif platform.system() == 'Linux':
                # Try to detect NVIDIA GPU on Linux
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                cuda_available = result.returncode == 0
                if cuda_available:
                    print("NVIDIA GPU detected, building with CUDA support")
                else:
                    print("No NVIDIA GPU detected, building without CUDA support")
        except Exception as e:
            print(f"Error checking for CUDA: {e}")
            print("Building without CUDA support")
        
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DCMAKE_INSTALL_PREFIX=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DCUTWED_BUILD_PYTHON=ON',
            f'-DCUTWED_USE_CUDA={str(cuda_available).upper()}',
        ]

        # Pass various options to CMake
        if 'CMAKE_ARGS' in os.environ:
            cmake_args += [item for item in os.environ['CMAKE_ARGS'].split(' ') if item]

        # Set build type
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        
        # Set generator
        if platform.system() == 'Windows':
            cmake_args += ['-G', 'Visual Studio 16 2019']
        else:
            cmake_args += ['-G', 'Ninja']
        
        build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
            
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--install', '.'], cwd=self.build_temp)


setup(
    name='cutwed',
    version='3.0.0',
    description='A linear memory CUDA algorithm for Time Warp Edit Distance with multiple backends',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Garrett Wright, cuTWED Contributors',
    author_email='garrett@gestaltgp.com',
    url='https://github.com/garrettwrong/cuTWED',
    packages=find_packages('python'),
    package_dir={'': 'python'},
    ext_modules=[CMakeExtension('cutwed')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)