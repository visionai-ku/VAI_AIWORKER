#include <nanobind/nanobind.h>

#ifdef RI_NATIVE_ENABLE_MUJOCO_ECHO
#include "mujoco_echo_bindings.hpp"
#endif

#ifdef RI_NATIVE_ENABLE_SIGNED_DISTANCE
#include "signed_distance_bindings.hpp"
#endif

#ifdef RI_NATIVE_ENABLE_IK
#include "ik_bindings.hpp"
#endif

#ifdef RI_NATIVE_ENABLE_TUTORIAL
#include "tutorial_bindings.hpp"
#endif

namespace nb = nanobind;

#ifndef RI_NATIVE_MODULE_NAME
#define RI_NATIVE_MODULE_NAME _mujoco_native
#endif

#define RI_NATIVE_STRINGIFY_INNER(x) #x
#define RI_NATIVE_STRINGIFY(x) RI_NATIVE_STRINGIFY_INNER(x)

nb::dict native_info() {
    nb::dict out;
    out["backend"] = "nanobind";
    out["module"] = RI_NATIVE_STRINGIFY(RI_NATIVE_MODULE_NAME);
    return out;
}

NB_MODULE(RI_NATIVE_MODULE_NAME,m) {
    m.doc() = "Native backend for ri_motion_v5_package.mujoco_sim.";
    m.def("native_info",&native_info,"Return native backend build information.");

#ifdef RI_NATIVE_ENABLE_TUTORIAL
    bind_tutorial(m);
#endif

#ifdef RI_NATIVE_ENABLE_MUJOCO_ECHO
    bind_mujoco_echo(m);
#endif

#ifdef RI_NATIVE_ENABLE_SIGNED_DISTANCE
    bind_signed_distance(m);
#endif

#ifdef RI_NATIVE_ENABLE_IK
    bind_ik(m);
#endif
}
