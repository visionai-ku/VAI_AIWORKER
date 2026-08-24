#include "mujoco_echo.hpp"

#include <cstring>
#include <vector>

#include <nanobind/ndarray.h>

#include "mujoco_handles.hpp"

namespace {

template <typename T>
nb::object copy_array(const T* src,std::initializer_list<size_t> shape) {
    /*
    MuJoCo owns arrays such as data->qpos, data->qvel, and data->xpos.
    Returning a NumPy view into those arrays would expose MuJoCo's internal
    memory directly to Python, and the view could outlive the mjData object
    that owns the memory.  For this echo test we want a safer and simpler
    contract: native code reads MuJoCo memory, copies the selected values into
    a C++ vector, and returns a NumPy array that owns that copy.

    nanobind needs a Python-visible owner object for the copied memory.  The
    capsule below owns the heap-allocated vector and deletes it when the NumPy
    array is garbage-collected.  This keeps the returned array independent of
    MuJoCo's lifetime while still avoiding an extra Python-side copy.
    */
    size_t count = 1;
    for (size_t dim : shape) {
        count *= dim;
    }

    auto* buffer = new std::vector<T>(count);
    if (count > 0) {
        std::memcpy(buffer->data(),src,sizeof(T)*count);
    }

    nb::capsule owner(buffer,[](void* p) noexcept {
        delete static_cast<std::vector<T>*>(p);
    });
    return nb::cast(nb::ndarray<nb::numpy,T>(buffer->data(),shape,owner));
}

}

nb::dict echo_model_data(std::uintptr_t model_address,std::uintptr_t data_address) {
    const mjModel* model = as_model(model_address);
    const mjData* data = as_data(data_address);

    nb::dict out;
    out["nq"] = model->nq;
    out["nv"] = model->nv;
    out["nu"] = model->nu;
    out["nbody"] = model->nbody;
    out["ngeom"] = model->ngeom;
    out["njnt"] = model->njnt;
    out["time"] = data->time;
    out["qpos"] = copy_array<double>(data->qpos,{static_cast<size_t>(model->nq)});
    out["qvel"] = copy_array<double>(data->qvel,{static_cast<size_t>(model->nv)});
    out["ctrl"] = copy_array<double>(data->ctrl,{static_cast<size_t>(model->nu)});
    out["body_xpos"] = copy_array<double>(
        data->xpos,{static_cast<size_t>(model->nbody),3}
    );
    out["geom_xpos"] = copy_array<double>(
        data->geom_xpos,{static_cast<size_t>(model->ngeom),3}
    );
    out["geom_bodyid"] = copy_array<int>(
        model->geom_bodyid,{static_cast<size_t>(model->ngeom)}
    );
    out["body_parentid"] = copy_array<int>(
        model->body_parentid,{static_cast<size_t>(model->nbody)}
    );
    out["geom_rbound"] = copy_array<double>(
        model->geom_rbound,{static_cast<size_t>(model->ngeom)}
    );

    return out;
}
