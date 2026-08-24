#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bind_mujoco_echo(nb::module_& m);
