#include "tutorial_bindings.hpp"

#include "tutorial.hpp"

void bind_tutorial(nb::module_& m) {
    m.def("foo",&foo,"Return value + 1.");
    m.def("boo",&boo,"Return value - 1.");
}
