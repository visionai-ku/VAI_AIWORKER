#include "mujoco_echo_bindings.hpp"

#include "mujoco_echo.hpp"

void bind_mujoco_echo(nb::module_& m) {
    m.def(
        "echo_model_data",
        &echo_model_data,
        "Echo selected mjModel and mjData fields from native code."
    );
}
