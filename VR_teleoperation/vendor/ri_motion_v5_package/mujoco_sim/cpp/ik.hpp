#pragma once

#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

nb::dict solve_body_position_ik(
    std::uintptr_t model_address,
    std::uintptr_t data_address,
    nb::ndarray<nb::numpy,const int,nb::ndim<1>,nb::c_contig> body_idxs,
    nb::ndarray<nb::numpy,const double,nb::ndim<2>,nb::c_contig> p_local,
    nb::ndarray<nb::numpy,const double,nb::ndim<2>,nb::c_contig> p_trgt,
    nb::ndarray<nb::numpy,const double,nb::ndim<1>,nb::c_contig> weights,
    nb::ndarray<nb::numpy,const int,nb::ndim<1>,nb::c_contig> rev_pri_qpos_idxs,
    nb::ndarray<nb::numpy,const int,nb::ndim<1>,nb::c_contig> idxs_jac_rev_pri,
    nb::ndarray<nb::numpy,const int,nb::ndim<1>,nb::c_contig> idxs_rev_pri_use,
    nb::ndarray<nb::numpy,const int,nb::ndim<1>,nb::c_contig> idxs_rev_pri_active,
    nb::ndarray<nb::numpy,const int,nb::ndim<1>,nb::c_contig> is_rev_in_rev_pri,
    nb::ndarray<nb::numpy,const double,nb::ndim<1>,nb::c_contig> rev_pri_joint_mins,
    nb::ndarray<nb::numpy,const double,nb::ndim<1>,nb::c_contig> rev_pri_joint_maxs,
    nb::ndarray<nb::numpy,const double,nb::ndim<1>,nb::c_contig> q_home_rev_pri,
    nb::ndarray<nb::numpy,const double,nb::ndim<1>,nb::c_contig> dls_damping_use,
    nb::ndarray<nb::numpy,const int,nb::ndim<1>,nb::c_contig> coupling_passive_idxs,
    nb::ndarray<nb::numpy,const int,nb::ndim<1>,nb::c_contig> coupling_active_idxs,
    nb::ndarray<nb::numpy,const double,nb::ndim<2>,nb::c_contig> coupling_coefs,
    nb::ndarray<nb::numpy,const double,nb::ndim<1>,nb::c_contig> coupling_x0,
    nb::ndarray<nb::numpy,const double,nb::ndim<1>,nb::c_contig> coupling_y0,
    int max_ik_tick,
    double ik_stepsize_rev,
    double ik_stepsize_pri,
    double ik_update_th_rev,
    double ik_update_th_pri,
    double max_probe_rev,
    double max_probe_pri,
    double k_null,
    bool joint_limit_handle_flag,
    bool nullspace_control_flag,
    bool task_null_mask_flag
);
