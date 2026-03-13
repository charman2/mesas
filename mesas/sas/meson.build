py.install_sources(
  '__init__.py',
  'functions.py',
  'model.py',
  'specs.py',
  subdir: 'mesas/sas',
)

# Build the f2py Fortran extension module
incdir_numpy = run_command(
  py, ['-c', 'import numpy; print(numpy.get_include())'],
  check: true,
).stdout().strip()

incdir_f2py = run_command(
  py, ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
  check: true,
).stdout().strip()

inc_np = include_directories(incdir_numpy)
inc_f2py = include_directories(incdir_f2py)

fortranobject_c = incdir_f2py / 'fortranobject.c'

# Generate the f2py wrapper C source from solve.f90.
# f2py produces solvemodule.c (the C wrapper) and solve-f2pywrappers.f
# (Fortran 77 wrappers, often empty for subroutine-only sources).
solve_f2py_wrapper = custom_target(
  'solvemodule.c',
  input: 'solve.f90',
  output: ['solvemodule.c', 'solve-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'solve', '--lower',
            '--build-dir', '@OUTDIR@'],
)

fc = meson.get_compiler('fortran')
fc_args = ['-Ofast', '-fPIC', '-fno-stack-arrays']

# Check which flags the compiler actually accepts
validated_fc_args = []
foreach arg : fc_args
  if fc.has_argument(arg)
    validated_fc_args += arg
  endif
endforeach

py.extension_module(
  'solve',
  ['solve.f90', fortranobject_c, solve_f2py_wrapper],
  include_directories: [inc_np, inc_f2py],
  fortran_args: validated_fc_args,
  install: true,
  subdir: 'mesas/sas',
)
