#pragma once

#include <cstdint>

#include <nanobind/nanobind.h>

namespace nb = nanobind;

nb::dict echo_model_data(std::uintptr_t model_address,std::uintptr_t data_address);
