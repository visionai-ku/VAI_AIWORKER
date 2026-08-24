# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/root/ros2_ws/src/cyclo_control/osqp_eigen_vendor/third_party/osqp-eigen")
  file(MAKE_DIRECTORY "/root/ros2_ws/src/cyclo_control/osqp_eigen_vendor/third_party/osqp-eigen")
endif()
file(MAKE_DIRECTORY
  "/root/ros2_ws/build/osqp_eigen_vendor/osqp_eigen_vendor-prefix/src/osqp_eigen_vendor-build"
  "/root/ros2_ws/build/osqp_eigen_vendor/osqp_eigen_vendor-prefix/install"
  "/root/ros2_ws/build/osqp_eigen_vendor/osqp_eigen_vendor-prefix/tmp"
  "/root/ros2_ws/build/osqp_eigen_vendor/osqp_eigen_vendor-prefix/src/osqp_eigen_vendor-stamp"
  "/root/ros2_ws/build/osqp_eigen_vendor/osqp_eigen_vendor-prefix/src"
  "/root/ros2_ws/build/osqp_eigen_vendor/osqp_eigen_vendor-prefix/src/osqp_eigen_vendor-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/root/ros2_ws/build/osqp_eigen_vendor/osqp_eigen_vendor-prefix/src/osqp_eigen_vendor-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/root/ros2_ws/build/osqp_eigen_vendor/osqp_eigen_vendor-prefix/src/osqp_eigen_vendor-stamp${cfgdir}") # cfgdir has leading slash
endif()
