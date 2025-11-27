/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/mip/solver_stats.hpp>

#include <linear_programming/initial_scaling_strategy/initial_scaling.cuh>
#include <mip/problem/problem.cuh>
#include <mip/relaxed_lp/lp_state.cuh>

#include <utilities/models/cpufj_predictor/header.h>
#include <utilities/models/fj_predictor/header.h>
#include <utilities/models/pdlp_predictor/header.h>
#include <utilities/work_limit_timer.hpp>
#include <utilities/work_unit_predictor.hpp>

#pragma once

// Forward declare
namespace cuopt::linear_programming::dual_simplex {
template <typename i_t, typename f_t>
class branch_and_bound_t;
}

namespace cuopt::linear_programming::detail {

struct mip_solver_work_unit_predictors_t {
  work_unit_predictor_t<fj_predictor> fj_predictor{};
  work_unit_predictor_t<cpufj_predictor> cpufj_predictor{};
  work_unit_predictor_t<pdlp_predictor> pdlp_predictor{};
};

// Aggregate structure containing the global context of the solving process for convenience:
// The current problem, user settings, raft handle and statistics objects
template <typename i_t, typename f_t>
struct mip_solver_context_t {
  explicit mip_solver_context_t(raft::handle_t const* handle_ptr_,
                                problem_t<i_t, f_t>* problem_ptr_,
                                mip_solver_settings_t<i_t, f_t> settings_,
                                pdlp_initial_scaling_strategy_t<i_t, f_t>& scaling)
    : handle_ptr(handle_ptr_), problem_ptr(problem_ptr_), settings(settings_), scaling(scaling)
  {
    cuopt_assert(problem_ptr != nullptr, "problem_ptr is nullptr");
    stats.solution_bound        = problem_ptr->maximize ? std::numeric_limits<f_t>::infinity()
                                                        : -std::numeric_limits<f_t>::infinity();
    gpu_heur_loop.deterministic = settings.deterministic;
  }

  raft::handle_t const* const handle_ptr;
  problem_t<i_t, f_t>* problem_ptr;
  dual_simplex::branch_and_bound_t<i_t, f_t>* branch_and_bound_ptr{nullptr};
  const mip_solver_settings_t<i_t, f_t> settings;
  pdlp_initial_scaling_strategy_t<i_t, f_t>& scaling;
  solver_stats_t<i_t, f_t> stats;
  // TODO: ensure thread local (or use locks...?)
  mip_solver_work_unit_predictors_t work_unit_predictors;
  // Work limit context for tracking work units in deterministic mode (shared across all timers in
  // GPU heuristic loop)
  cuopt::work_limit_context_t gpu_heur_loop;
};

}  // namespace cuopt::linear_programming::detail
