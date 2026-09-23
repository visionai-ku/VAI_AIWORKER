#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "robotis_hand_pressure_broadcaster::robotis_hand_pressure_broadcaster" for configuration "Release"
set_property(TARGET robotis_hand_pressure_broadcaster::robotis_hand_pressure_broadcaster APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(robotis_hand_pressure_broadcaster::robotis_hand_pressure_broadcaster PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/librobotis_hand_pressure_broadcaster.so"
  IMPORTED_SONAME_RELEASE "librobotis_hand_pressure_broadcaster.so"
  )

list(APPEND _cmake_import_check_targets robotis_hand_pressure_broadcaster::robotis_hand_pressure_broadcaster )
list(APPEND _cmake_import_check_files_for_robotis_hand_pressure_broadcaster::robotis_hand_pressure_broadcaster "${_IMPORT_PREFIX}/lib/librobotis_hand_pressure_broadcaster.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
