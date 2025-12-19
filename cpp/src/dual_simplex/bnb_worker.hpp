/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/bounds_strengthening.hpp>
#include <dual_simplex/mip_node.hpp>
#include <dual_simplex/phase2.hpp>

#include <vector>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
struct bnb_stats_t {
  f_t start_time                        = 0.0;
  omp_atomic_t<f_t> total_lp_solve_time = 0.0;
  omp_atomic_t<i_t> nodes_explored      = 0;
  omp_atomic_t<i_t> nodes_unexplored    = 0;
  omp_atomic_t<f_t> total_lp_iters      = 0;
};

template <typename i_t, typename f_t>
class bnb_worker_t {
 public:
  const i_t worker_id;
  omp_atomic_t<bnb_worker_type_t> worker_type;
  omp_atomic_t<bool> is_active;
  omp_atomic_t<f_t> lower_bound;

  lp_problem_t<i_t, f_t> leaf_problem;

  basis_update_mpf_t<i_t, f_t> basis_factors;
  std::vector<i_t> basic_list;
  std::vector<i_t> nonbasic_list;

  bounds_strengthening_t<i_t, f_t> node_presolver;
  std::vector<bool> bounds_changed;

  std::vector<f_t> start_lower;
  std::vector<f_t> start_upper;
  mip_node_t<i_t, f_t>* start_node;

  bool recompute_basis  = true;
  bool recompute_bounds = true;

  bnb_worker_t(i_t worker_id,
               const lp_problem_t<i_t, f_t>& original_lp,
               const csr_matrix_t<i_t, f_t>& Arow,
               const std::vector<variable_type_t>& var_type,
               const simplex_solver_settings_t<i_t, f_t>& settings);

  // Set the `start_node` for best-first search.
  void init_best_first(mip_node_t<i_t, f_t>* node, const lp_problem_t<i_t, f_t>& original_lp)
  {
    start_node  = node;
    start_lower = original_lp.lower;
    start_upper = original_lp.upper;
    worker_type = EXPLORATION;
    lower_bound = node->lower_bound;
    is_active   = true;
  }

  // Initialize the worker for diving, setting the `start_node`, `start_lower` and
  // `start_upper`. Returns `true` if the starting node is feasible via
  // bounds propagation.
  bool init_diving(mip_node_t<i_t, f_t>* node,
                   bnb_worker_type_t type,
                   const lp_problem_t<i_t, f_t>& original_lp,
                   const simplex_solver_settings_t<i_t, f_t>& settings);

  // Set the variables bounds for the LP relaxation of the current node.
  bool set_lp_variable_bounds_for(mip_node_t<i_t, f_t>* node_ptr,
                                  const simplex_solver_settings_t<i_t, f_t>& settings);

 private:
  // For diving, we need to store the full node instead of
  // of just a pointer, since it is detached from the
  // tree. To keep the same interface for any type of worker,
  // the start node will point to this node when diving.
  // For best-first search, this will not be used.
  mip_node_t<i_t, f_t> internal_node;
};

}  // namespace cuopt::linear_programming::dual_simplex
