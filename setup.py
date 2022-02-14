from setuptools import setup, Extension, distutils, Command, find_packages
import setuptools.command.build_ext
import setuptools.command.install
import distutils.command.build
import distutils.command.clean
import platform
import subprocess
import shutil
import sys
import os

# TODO: make this more robust
WITH_CUDA = os.path.exists('/Developer/NVIDIA/CUDA-7.5/include') or os.path.exists('/usr/local/cuda/include')
DEBUG = False

################################################################################
# Monkey-patch setuptools to compile in parallel
################################################################################

# Monkey-patch 虽然这方法并不是类中的方法，但是为了monkey-patch,第一个参数仍为self
# 通过multiprocessing 来实现parallel
# cpu_count来决定并行数
def parallelCCompile(self, sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
    # those lines are copied from distutils.ccompiler.CCompiler directly
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(output_dir, macros, include_dirs, sources, depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    # compile using a thread pool
    import multiprocessing.pool
    def _single_compile(obj):
        src, ext = build[obj]
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
    num_jobs = multiprocessing.cpu_count()
    multiprocessing.pool.ThreadPool(num_jobs).map(_single_compile, objects)

    return objects

distutils.ccompiler.CCompiler.compile = parallelCCompile

################################################################################
# Custom build commands
################################################################################

class build_deps(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # from tools.nnwrap import generate_wrappers as generate_nn_wrappers
        build_all_cmd = ['bash', 'torch/lib/build_all.sh']
        # if WITH_CUDA:
        #     build_all_cmd += ['--with-cuda']
        if subprocess.call(build_all_cmd) != 0:
            sys.exit(1)
        # generate_nn_wrappers()

setup(name="torch", version="0.1",
    # ext_modules=extensions,
    cmdclass = {
        # 'build': build,
        # 'build_ext': build_ext,
        'build_deps': build_deps,
        # 'build_module': build_module,
        # 'install': install,
        # 'clean': clean,
    },
    # packages=packages,
    # package_data={'torch': ['lib/*.so*', 'lib/*.h']},
    # install_requires=['pyyaml'],
)