if(NOT EXISTS "/root/ros2_ws/build/osqp_eigen_vendor/osqp_eigen_vendor-prefix/src/osqp_eigen_vendor-build/install_manifest.txt")
  message(WARNING "Cannot find install manifest: \"/root/ros2_ws/build/osqp_eigen_vendor/osqp_eigen_vendor-prefix/src/osqp_eigen_vendor-build/install_manifest.txt\"")
  return()
endif()

file(READ "/root/ros2_ws/build/osqp_eigen_vendor/osqp_eigen_vendor-prefix/src/osqp_eigen_vendor-build/install_manifest.txt" files)
string(STRIP "${files}" files)
string(REGEX REPLACE "\n" ";" files "${files}")
list(REVERSE files)
foreach(file ${files})
  if(IS_SYMLINK "$ENV{DESTDIR}${file}" OR EXISTS "$ENV{DESTDIR}${file}")
    message(STATUS "Uninstalling: $ENV{DESTDIR}${file}")
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E remove "$ENV{DESTDIR}${file}"
      OUTPUT_VARIABLE rm_out
      RESULT_VARIABLE rm_retval)
    if(NOT "${rm_retval}" EQUAL 0)
      message(FATAL_ERROR "Problem when removing \"$ENV{DESTDIR}${file}\"")
    endif()
  else()
    message(STATUS "Not-found: $ENV{DESTDIR}${file}")
  endif()
endforeach()
