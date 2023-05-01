cmake_minimum_required(VERSION 3.0)

project(mesas)
enable_language(Fortran)
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/mod)
include_directories(${CMAKE_Fortran_MODULE_DIRECTORY})
add_compile_options(-fast)


add_library(biomath_constants_mod STATIC biomath_constants_mod.f90)
add_library(biomath_sort_mod
  STATIC
    biomath_sort_mod.f90
    biomath_constants_mod.f90
)
target_link_libraries(biomath_sort_mod biomath_constants_mod)
add_library(biomath_strings_mod
  STATIC
    biomath_strings_mod.f90
    biomath_constants_mod.f90
)
target_link_libraries(biomath_strings_mod biomath_constants_mod)
add_library(biomath_interface_mod
  STATIC
    biomath_interface_mod.f90
    biomath_constants_mod.f90
    biomath_sort_mod.f90
    biomath_strings_mod.f90
)
target_link_libraries(biomath_interface_mod biomath_strings_mod biomath_sort_mod)
add_library(biomath_mathlib_mod
  STATIC
    biomath_mathlib_mod.f90
    biomath_constants_mod.f90
)
target_link_libraries(biomath_mathlib_mod biomath_constants_mod)
add_library(zero_finder
  STATIC
    zero_finder.f90
    biomath_constants_mod.f90
)
target_link_libraries(zero_finder biomath_constants_mod)
add_library(cdf_aux_mod
  STATIC
    cdf_aux_mod.f90
    biomath_constants_mod.f90
    biomath_interface_mod.f90
    zero_finder.f90
    biomath_strings_mod.f90
    biomath_sort_mod.f90
)
target_link_libraries(cdf_aux_mod biomath_interface_mod zero_finder)
add_library(cdf_beta_mod
  STATIC
    cdf_beta_mod.f90
    biomath_constants_mod.f90
    biomath_mathlib_mod.f90
    cdf_aux_mod.f90
    zero_finder.f90
)
target_link_libraries(cdf_beta_mod biomath_mathlib_mod cdf_aux_mod)
add_library(cdf_binomial_mod
  STATIC
    cdf_binomial_mod.f90
    biomath_constants_mod.f90
    cdf_aux_mod.f90
    cdf_beta_mod.f90
    zero_finder.f90
    biomath_mathlib_mod.f90
)
target_link_libraries(cdf_binomial_mod cdf_beta_mod)
add_library(cdf_gamma_mod
  STATIC
    cdf_gamma_mod.f90
    biomath_constants_mod.f90
    biomath_mathlib_mod.f90
    cdf_aux_mod.f90
    zero_finder.f90
)
target_link_libraries(cdf_gamma_mod biomath_mathlib_mod cdf_aux_mod)
add_library(cdf_chisq_mod
  STATIC
    cdf_chisq_mod.f90
    biomath_constants_mod.f90
    cdf_aux_mod.f90
    cdf_gamma_mod.f90
    zero_finder.f90
    biomath_mathlib_mod.f90
)
target_link_libraries(cdf_chisq_mod cdf_gamma_mod)
add_library(cdf_f_mod
  STATIC
    cdf_f_mod.f90
    biomath_constants_mod.f90
    cdf_aux_mod.f90
    cdf_beta_mod.f90
    zero_finder.f90
    biomath_mathlib_mod.f90
)
target_link_libraries(cdf_f_mod biomath_mathlib_mod cdf_aux_mod)
add_library(cdf_nc_chisq_mod
  STATIC
    cdf_nc_chisq_mod.f90
    biomath_constants_mod.f90
    biomath_mathlib_mod.f90
    cdf_aux_mod.f90
    cdf_chisq_mod.f90
    zero_finder.f90
    cdf_gamma_mod.f90
)
target_link_libraries(cdf_nc_chisq_mod cdf_chisq_mod)
add_library(cdf_nc_f_mod
  STATIC
    cdf_nc_f_mod.f90
    biomath_constants_mod.f90
    biomath_mathlib_mod.f90
    cdf_aux_mod.f90
    cdf_beta_mod.f90
    cdf_f_mod.f90
    cdf_gamma_mod.f90
    zero_finder.f90
)
target_link_libraries(cdf_nc_f_mod cdf_f_mod)
add_library(cdf_normal_mod
  STATIC
    cdf_normal_mod.f90
    biomath_constants_mod.f90
    biomath_mathlib_mod.f90
    cdf_aux_mod.f90
    zero_finder.f90
)
target_link_libraries(cdf_normal_mod biomath_mathlib_mod cdf_aux_mod)
add_library(cdf_t_mod
  STATIC
    cdf_t_mod.f90
    biomath_constants_mod.f90
    biomath_mathlib_mod.f90
    cdf_aux_mod.f90
    cdf_beta_mod.f90
    cdf_normal_mod.f90
    zero_finder.f90
)
target_link_libraries(cdf_t_mod cdf_normal_mod cdf_beta_mod)
add_library(cdf_nc_t_mod
  STATIC
    cdf_nc_t_mod.f90
    biomath_constants_mod.f90
    biomath_mathlib_mod.f90
    cdf_aux_mod.f90
    cdf_beta_mod.f90
    cdf_normal_mod.f90
    cdf_t_mod.f90
    zero_finder.f90
)
target_link_libraries(cdf_nc_t_mod cdf_t_mod)
add_library(cdf_neg_binomial_mod
  STATIC
    cdf_neg_binomial_mod.f90
    biomath_constants_mod.f90
    cdf_aux_mod.f90
    cdf_beta_mod.f90
    zero_finder.f90
    biomath_mathlib_mod.f90
)
target_link_libraries(cdf_neg_binomial_mod cdf_beta_mod)
add_library(cdf_poisson_mod
  STATIC
    cdf_poisson_mod.f90
    biomath_constants_mod.f90
    cdf_aux_mod.f90
    cdf_gamma_mod.f90
    biomath_mathlib_mod.f90
    zero_finder.f90
)
target_link_libraries(cdf_poisson_mod cdf_gamma_mod)


install(
  TARGETS
    biomath_constants_mod
    biomath_sort_mod
    biomath_strings_mod
    biomath_interface_mod
    biomath_mathlib_mod
    zero_finder
    cdf_aux_mod
    cdf_beta_mod
    cdf_binomial_mod
    cdf_gamma_mod
    cdf_chisq_mod
    cdf_f_mod
    cdf_nc_chisq_mod
    cdf_nc_f_mod
    cdf_normal_mod
    cdf_t_mod
    cdf_nc_t_mod
    cdf_neg_binomial_mod
    cdf_poisson_mod
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

file(GLOB MOD_FILES ${CMAKE_Fortran_MODULE_DIRECTORY}/*.mod)
install(FILES ${MOD_FILES} DESTINATION include)
