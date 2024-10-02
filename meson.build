project('mesas_solve', 'fortran')

python = import('python').find_installation(pure: false)
f2py = find_program('f2py')

py_ext_suffix = run_command(
  python, '-c', 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))'
).stdout().strip()
file_extension = py_ext_suffix.split('.')[-1]

if host_machine.system() == 'windows'
  rename_command = [
    'powershell', '-Command',
    'Rename-Item', '@INPUT@', 'mesas_solve.' + file_extension
  ]
else
  rename_command = ['mv', '@INPUT@', 'mesas_solve.' + file_extension]
endif

fortran_extension = custom_target(
  'mesas_solve',
  input: files('mesas/sas/solve.f90'),
  output: 'mesas_solve' + py_ext_suffix,
  command: [f2py, '-c', '-m', 'mesas_solve', '@INPUT@'],
  install: false
)

rename_mesas_solve = custom_target(
  'rename_mesas_solve',
  input: fortran_extension,
  output: 'mesas_solve.' + file_extension,
  command: rename_command,
  install: true,
  install_dir: python.get_install_dir()
)

install_subdir('mesas', install_dir: python.get_install_dir())
