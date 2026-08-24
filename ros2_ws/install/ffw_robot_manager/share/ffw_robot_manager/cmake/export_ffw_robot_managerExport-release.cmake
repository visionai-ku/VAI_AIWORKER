#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ffw_robot_manager::ffw_robot_manager" for configuration "Release"
set_property(TARGET ffw_robot_manager::ffw_robot_manager APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ffw_robot_manager::ffw_robot_manager PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libffw_robot_manager.so"
  IMPORTED_SONAME_RELEASE "libffw_robot_manager.so"
  )

list(APPEND _cmake_import_check_targets ffw_robot_manager::ffw_robot_manager )
list(APPEND _cmake_import_check_files_for_ffw_robot_manager::ffw_robot_manager "${_IMPORT_PREFIX}/lib/libffw_robot_manager.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
