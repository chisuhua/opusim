project(
  'isasim',
  ['c', 'cpp'],
  version : run_command(
    [find_program('python', 'python3'), 'bin/meson_get_version.py']
  ).stdout(),
  license : 'MIT',
  meson_version : '>= 0.52',
  default_options : ['buildtype=debug'] #, 'b_ndebug=if-release', 'c_std=c99', 'cpp_std=c++17']
)

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')
buildtype = get_option('buildtype')
builddir = meson.current_build_dir()

null_dep = dependency('', required : false)

# Arguments for the preprocessor, put these in a separate array from the C and
# C++ (cpp in meson terminology) arguments since they need to be added to the
# default arguments for both C and C++.
pre_args = [
  '-D__STDC_CONSTANT_MACROS',
  '-D__STDC_FORMAT_MACROS',
  '-D__STDC_LIMIT_MACROS',
  '-DPACKAGE_VERSION="@0@"'.format(meson.project_version()),
  '-DPACKAGE_BUGREPORT="https://gitlab.freedesktop.org/mesa/mesa/-/issues"',
]

extra_cpp_args = ['-Wall', '-std=c++17']

if buildtype == 'release'
  extra_cpp_args += ['-O3', '-g', '-Wno-strict-aliasing', '-msse4.1', '-flto']
else
  extra_cpp_args += ['-O0', '-g', '-Wno-strict-aliasing', '-msse4.1']
endif

extra_c_args = extra_cpp_args

#if cpp.get_id() == 'gcc'
#  extra_cpp_args += ['-std=c++17']
#else
#  extra_cpp_args += []
#endif

link_args = ['-ldl', '-lpthread', '-L' + meson.current_build_dir(), '-lstdc++' ]

model_symbol_list = meson.current_source_dir() + '/model_symbol_export.def'

#pymodule = import('python3')
#python3 = pymodule.find_python()

DESIGN_ROOT = run_command([find_program('python', 'python3'), '-c', 'import os; print(os.environ["DESIGN_ROOT"])']).stdout().strip()
GEM5_ROOT = run_command([find_program('python', 'python3'), '-c', 'import os; print(os.environ["GEM5_ROOT"])']).stdout().strip()
CUDA_INSTALL_PATH = run_command([find_program('python', 'python3'), '-c', 'import os; print(os.environ["CUDA_INSTALL_PATH"])']).stdout().strip()

message('DESIGN_ROOT is ' + DESIGN_ROOT)
message('GEM5_ROOT is ' + GEM5_ROOT)
message('CUDA_INSTALL_PATH is ' + CUDA_INSTALL_PATH)

isa_link_args = link_args + ['-L' + DESIGN_ROOT + '/cosim', '-lcudart', '-lcash' ] + ['-Wl,--version-script='+model_symbol_list]
#isa_link_args += ['-L' + DESIGN_ROOT + '/opu/3rdparty/cash/build/lib', '-lcash']
isa_inc = [
  'isa',
  'model',
  'hwop',
  DESIGN_ROOT + '/opu/coasm',
  DESIGN_ROOT + '/opu/umd',
  DESIGN_ROOT + '/libcuda',
  DESIGN_ROOT + '/opu/3rdparty/cash/include',
  DESIGN_ROOT + '/opu/3rdparty/cash/soft-ieee754/includes',
  CUDA_INSTALL_PATH + '/include'
  ]

isa_src = [
  'common/string_utils.cpp',
  'common/debug.cpp',
  'model/Executor.cpp',
  'model/Kernel.cpp',
  'model/Warp.cpp',
  'model/CUResource.cpp',
  'model/WarpState.cpp',
  'model/BlockState.cpp',
  'model/ThreadBlock.cpp',
  'model/ThreadItem.cpp',
  #'isa/Compute.cpp',
  #'isa/ComputeUnit.cpp',
  #'isa/CopyEngine.cpp',
  #'isa/StreamProcessor.cpp',
  #'isa/Engine.cpp',
  #'isa/Memory.cpp',
  #'isa/MemoryImpl.cc',
  #'isa/Mmio.cpp',
  #'isa/Warp.cpp',
  'hwop/IAlu.cpp',
  'hwop/FunUnit.cpp',
  'isa/Instruction.cpp',
  'isa/InstructionDLS.cpp',
  'isa/InstructionSLS.cpp',
  'isa/InstructionVLS.cpp',
  #'isa/InstructionMTBUF.cpp',
  'isa/InstructionVMUBUF.cpp',
  #'isa/InstructionSMRD.cpp',
  'isa/InstructionSOP1.cpp',
  'isa/InstructionSOP2.cpp',
  'isa/InstructionSOPC.cpp',
  'isa/InstructionSOPK.cpp',
  'isa/InstructionSOPP.cpp',
  'isa/InstructionVOP1.cpp',
  'isa/InstructionVOP2.cpp',
  'isa/InstructionVOP3A.cpp',
  'isa/InstructionVOP3B.cpp',
  'isa/InstructionVOPC.cpp',
  'isa/DecodeCommon.cpp',
  'isa/ExecCommon.cpp',
  'isa/MemCommon.cpp'
  ]

isa = shared_library(
  'isasim',
  isa_src,
  include_directories : include_directories(isa_inc),
  install : false,
  cpp_args : extra_cpp_args,
  link_args : isa_link_args,
  )


#test_simple_src = [
#  'test/main.cpp',
#  'test/module/Simple.cpp'
#  ]

#test_simple = executable(
#  'test_simple',
#  test_simple_src,
#  include_directories : include_directories(module_inc),
#  install : false,
#  cpp_args : extra_cpp_args,
#  link_args : module_link_args,
#  #dependencies: declare_dependency(link_with : )
#  )

