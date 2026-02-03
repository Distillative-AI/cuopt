/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/bnb_worker.hpp>
#include <dual_simplex/logger.hpp>
#include <dual_simplex/mip_node.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/types.hpp>
#include <utilities/omp_helpers.hpp>
#include <utilities/pcg.hpp>

#include <omp.h>
#include <cmath>
#include <cstdint>

namespace cuopt::linear_programming::dual_simplex {

// =============================================================================
// Pseudo-cost utility functions (lock-free implementations)
// These can be called directly with snapshot data or through pseudo_costs_t methods
// =============================================================================

template <typename f_t>
struct pseudo_cost_averages_t {
  f_t down_avg;
  f_t up_avg;
};

// Compute average pseudo-costs from arrays
// Works with either pseudo_costs_t members or snapshot arrays
template <typename i_t, typename f_t>
pseudo_cost_averages_t<f_t> compute_pseudo_cost_averages(
  const f_t* pc_sum_down, const f_t* pc_sum_up, const i_t* pc_num_down, const i_t* pc_num_up, i_t n)
{
  i_t num_initialized_down = 0;
  i_t num_initialized_up   = 0;
  f_t pseudo_cost_down_avg = 0;
  f_t pseudo_cost_up_avg   = 0;

  for (i_t j = 0; j < n; ++j) {
    if (pc_num_down[j] > 0) {
      ++num_initialized_down;
      if (std::isfinite(pc_sum_down[j])) {
        pseudo_cost_down_avg += pc_sum_down[j] / pc_num_down[j];
      }
    }
    if (pc_num_up[j] > 0) {
      ++num_initialized_up;
      if (std::isfinite(pc_sum_up[j])) { pseudo_cost_up_avg += pc_sum_up[j] / pc_num_up[j]; }
    }
  }

  pseudo_cost_down_avg =
    (num_initialized_down > 0) ? pseudo_cost_down_avg / num_initialized_down : f_t(1.0);
  pseudo_cost_up_avg =
    (num_initialized_up > 0) ? pseudo_cost_up_avg / num_initialized_up : f_t(1.0);

  return {pseudo_cost_down_avg, pseudo_cost_up_avg};
}

// Variable selection using pseudo-cost product scoring
// Returns the best variable to branch on
template <typename i_t, typename f_t>
i_t variable_selection_from_pseudo_costs(const f_t* pc_sum_down,
                                         const f_t* pc_sum_up,
                                         const i_t* pc_num_down,
                                         const i_t* pc_num_up,
                                         i_t n_vars,
                                         const std::vector<i_t>& fractional,
                                         const std::vector<f_t>& solution)
{
  const i_t num_fractional = fractional.size();
  if (num_fractional == 0) return -1;

  auto [pc_down_avg, pc_up_avg] =
    compute_pseudo_cost_averages(pc_sum_down, pc_sum_up, pc_num_down, pc_num_up, n_vars);

  i_t branch_var    = fractional[0];
  f_t max_score     = std::numeric_limits<f_t>::lowest();
  constexpr f_t eps = f_t(1e-6);

  for (i_t j : fractional) {
    f_t pc_down      = pc_num_down[j] != 0 ? pc_sum_down[j] / pc_num_down[j] : pc_down_avg;
    f_t pc_up        = pc_num_up[j] != 0 ? pc_sum_up[j] / pc_num_up[j] : pc_up_avg;
    const f_t f_down = solution[j] - std::floor(solution[j]);
    const f_t f_up   = std::ceil(solution[j]) - solution[j];
    f_t score        = std::max(f_down * pc_down, eps) * std::max(f_up * pc_up, eps);
    if (score > max_score) {
      max_score  = score;
      branch_var = j;
    }
  }

  return branch_var;
}

template <typename i_t, typename f_t>
class pseudo_costs_t {
 public:
  explicit pseudo_costs_t(i_t num_variables)
    : pseudo_cost_sum_down(num_variables),
      pseudo_cost_sum_up(num_variables),
      pseudo_cost_num_down(num_variables),
      pseudo_cost_num_up(num_variables),
      pseudo_cost_mutex(num_variables)
  {
  }

  void update_pseudo_costs(mip_node_t<i_t, f_t>* node_ptr, f_t leaf_objective);

  void resize(i_t num_variables)
  {
    pseudo_cost_sum_down.assign(num_variables, 0);
    pseudo_cost_sum_up.assign(num_variables, 0);
    pseudo_cost_num_down.assign(num_variables, 0);
    pseudo_cost_num_up.assign(num_variables, 0);
    pseudo_cost_mutex.resize(num_variables);
  }

  void initialized(i_t& num_initialized_down,
                   i_t& num_initialized_up,
                   f_t& pseudo_cost_down_avg,
                   f_t& pseudo_cost_up_avg);

  f_t obj_estimate(const std::vector<i_t>& fractional,
                   const std::vector<f_t>& solution,
                   f_t lower_bound,
                   logger_t& log);

  i_t variable_selection(const std::vector<i_t>& fractional,
                         const std::vector<f_t>& solution,
                         logger_t& log);

  i_t reliable_variable_selection(mip_node_t<i_t, f_t>* node_ptr,
                                  const std::vector<i_t>& fractional,
                                  const std::vector<f_t>& solution,
                                  const simplex_solver_settings_t<i_t, f_t>& settings,
                                  const std::vector<variable_type_t>& var_types,
                                  bnb_worker_data_t<i_t, f_t>* worker_data,
                                  const bnb_stats_t<i_t, f_t>& bnb_stats,
                                  f_t upper_bound,
                                  logger_t& log);

  void update_pseudo_costs_from_strong_branching(const std::vector<i_t>& fractional,
                                                 const std::vector<f_t>& root_soln);

  uint32_t compute_state_hash() const
  {
    return detail::compute_hash(pseudo_cost_sum_down) ^ detail::compute_hash(pseudo_cost_sum_up) ^
           detail::compute_hash(pseudo_cost_num_down) ^ detail::compute_hash(pseudo_cost_num_up);
  }

  uint32_t compute_strong_branch_hash() const
  {
    return detail::compute_hash(strong_branch_down) ^ detail::compute_hash(strong_branch_up);
  }

  std::vector<omp_atomic_t<f_t>> pseudo_cost_sum_up;
  std::vector<omp_atomic_t<f_t>> pseudo_cost_sum_down;
  std::vector<omp_atomic_t<i_t>> pseudo_cost_num_up;
  std::vector<omp_atomic_t<i_t>> pseudo_cost_num_down;
  std::vector<f_t> strong_branch_down;
  std::vector<f_t> strong_branch_up;
  std::vector<omp_mutex_t> pseudo_cost_mutex;
  omp_atomic_t<i_t> num_strong_branches_completed = 0;
  omp_atomic_t<int64_t> sb_total_lp_iter          = 0;
};

template <typename i_t, typename f_t>
void strong_branching(const user_problem_t<i_t, f_t>& original_problem,
                      const lp_problem_t<i_t, f_t>& original_lp,
                      const simplex_solver_settings_t<i_t, f_t>& settings,
                      f_t start_time,
                      const std::vector<variable_type_t>& var_types,
                      const std::vector<f_t> root_soln,
                      const std::vector<i_t>& fractional,
                      f_t root_obj,
                      const std::vector<variable_status_t>& root_vstatus,
                      const std::vector<f_t>& edge_norms,
                      pseudo_costs_t<i_t, f_t>& pc);

}  // namespace cuopt::linear_programming::dual_simplex
