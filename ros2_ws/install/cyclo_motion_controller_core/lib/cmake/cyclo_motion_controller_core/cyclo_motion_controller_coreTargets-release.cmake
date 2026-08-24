#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "cyclo_motion_controller_core::cyclo_motion_controller_core" for configuration "Release"
set_property(TARGET cyclo_motion_controller_core::cyclo_motion_controller_core APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(cyclo_motion_controller_core::cyclo_motion_controller_core PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libcyclo_motion_controller_core.so"
  IMPORTED_SONAME_RELEASE "libcyclo_motion_controller_core.so"
  )

list(APPEND _cmake_import_check_targets cyclo_motion_controller_core::cyclo_motion_controller_core )
list(APPEND _cmake_import_check_files_for_cyclo_motion_controller_core::cyclo_motion_controller_core "${_IMPORT_PREFIX}/lib/libcyclo_motion_controller_core.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
