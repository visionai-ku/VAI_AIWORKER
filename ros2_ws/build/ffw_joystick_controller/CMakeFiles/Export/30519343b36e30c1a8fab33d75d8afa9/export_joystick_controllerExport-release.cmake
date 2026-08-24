#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ffw_joystick_controller::joystick_controller" for configuration "Release"
set_property(TARGET ffw_joystick_controller::joystick_controller APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ffw_joystick_controller::joystick_controller PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libjoystick_controller.so"
  IMPORTED_SONAME_RELEASE "libjoystick_controller.so"
  )

list(APPEND _cmake_import_check_targets ffw_joystick_controller::joystick_controller )
list(APPEND _cmake_import_check_files_for_ffw_joystick_controller::joystick_controller "${_IMPORT_PREFIX}/lib/libjoystick_controller.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
