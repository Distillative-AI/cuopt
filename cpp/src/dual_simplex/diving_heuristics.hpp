/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/bounds_strengthening.hpp>
#include <dual_simplex/pseudo_costs.hpp>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t>
struct branch_variable_t {
  i_t variable;
  rounding_direction_t direction;
};

// =============================================================================
// Lock-free diving implementations that work on raw pseudo-cost arrays
// These can be called directly with snapshot data or through the wrapper functions
// =============================================================================

// Pseudocost diving variable selection (lock-free implementation)
template <typename i_t, typename f_t>
branch_variable_t<i_t> pseudocost_diving_from_arrays(const f_t* pc_sum_down,
                                                     const f_t* pc_sum_up,
                                                     const i_t* pc_num_down,
                                                     const i_t* pc_num_up,
                                                     i_t n_vars,
                                                     const std::vector<i_t>& fractional,
                                                     const std::vector<f_t>& solution,
                                                     const std::vector<f_t>& root_solution)
{
  const i_t num_fractional = fractional.size();
  if (num_fractional == 0) return {-1, rounding_direction_t::NONE};

  auto avgs = compute_pseudo_cost_averages(pc_sum_down, pc_sum_up, pc_num_down, pc_num_up, n_vars);

  i_t branch_var                 = fractional[0];
  f_t max_score                  = std::numeric_limits<f_t>::lowest();
  rounding_direction_t round_dir = rounding_direction_t::DOWN;
  constexpr f_t eps              = f_t(1e-6);

  for (i_t j : fractional) {
    f_t f_down  = solution[j] - std::floor(solution[j]);
    f_t f_up    = std::ceil(solution[j]) - solution[j];
    f_t pc_down = pc_num_down[j] != 0 ? pc_sum_down[j] / pc_num_down[j] : avgs.down_avg;
    f_t pc_up   = pc_num_up[j] != 0 ? pc_sum_up[j] / pc_num_up[j] : avgs.up_avg;

    f_t score_down = std::sqrt(f_up) * (1 + pc_up) / (1 + pc_down);
    f_t score_up   = std::sqrt(f_down) * (1 + pc_down) / (1 + pc_up);

    f_t score                = 0;
    rounding_direction_t dir = rounding_direction_t::DOWN;

    f_t root_val = (j < static_cast<i_t>(root_solution.size())) ? root_solution[j] : solution[j];

    if (solution[j] < root_val - f_t(0.4)) {
      score = score_down;
      dir   = rounding_direction_t::DOWN;
    } else if (solution[j] > root_val + f_t(0.4)) {
      score = score_up;
      dir   = rounding_direction_t::UP;
    } else if (f_down < f_t(0.3)) {
      score = score_down;
      dir   = rounding_direction_t::DOWN;
    } else if (f_down > f_t(0.7)) {
      score = score_up;
      dir   = rounding_direction_t::UP;
    } else if (pc_down < pc_up + eps) {
      score = score_down;
      dir   = rounding_direction_t::DOWN;
    } else {
      score = score_up;
      dir   = rounding_direction_t::UP;
    }

    if (score > max_score) {
      max_score  = score;
      branch_var = j;
      round_dir  = dir;
    }
  }

  return {branch_var, round_dir};
}

// Guided diving variable selection (lock-free implementation)
template <typename i_t, typename f_t>
branch_variable_t<i_t> guided_diving_from_arrays(const f_t* pc_sum_down,
                                                 const f_t* pc_sum_up,
                                                 const i_t* pc_num_down,
                                                 const i_t* pc_num_up,
                                                 i_t n_vars,
                                                 const std::vector<i_t>& fractional,
                                                 const std::vector<f_t>& solution,
                                                 const std::vector<f_t>& incumbent)
{
  const i_t num_fractional = fractional.size();
  if (num_fractional == 0) return {-1, rounding_direction_t::NONE};

  auto avgs = compute_pseudo_cost_averages(pc_sum_down, pc_sum_up, pc_num_down, pc_num_up, n_vars);

  i_t branch_var                 = fractional[0];
  f_t max_score                  = std::numeric_limits<f_t>::lowest();
  rounding_direction_t round_dir = rounding_direction_t::DOWN;
  constexpr f_t eps              = f_t(1e-6);

  for (i_t j : fractional) {
    f_t f_down    = solution[j] - std::floor(solution[j]);
    f_t f_up      = std::ceil(solution[j]) - solution[j];
    f_t down_dist = std::abs(incumbent[j] - std::floor(solution[j]));
    f_t up_dist   = std::abs(std::ceil(solution[j]) - incumbent[j]);
    rounding_direction_t dir =
      down_dist < up_dist + eps ? rounding_direction_t::DOWN : rounding_direction_t::UP;

    f_t pc_down = pc_num_down[j] != 0 ? pc_sum_down[j] / pc_num_down[j] : avgs.down_avg;
    f_t pc_up   = pc_num_up[j] != 0 ? pc_sum_up[j] / pc_num_up[j] : avgs.up_avg;

    f_t score1 = dir == rounding_direction_t::DOWN ? 5 * pc_down * f_down : 5 * pc_up * f_up;
    f_t score2 = dir == rounding_direction_t::DOWN ? pc_up * f_up : pc_down * f_down;
    f_t score  = (score1 + score2) / 6;

    if (score > max_score) {
      max_score  = score;
      branch_var = j;
      round_dir  = dir;
    }
  }

  return {branch_var, round_dir};
}

// =============================================================================
// Wrapper functions that acquire locks and call the impl versions
// =============================================================================

template <typename i_t, typename f_t>
branch_variable_t<i_t> line_search_diving(const std::vector<i_t>& fractional,
                                          const std::vector<f_t>& solution,
                                          const std::vector<f_t>& root_solution,
                                          logger_t& log);

template <typename i_t, typename f_t>
branch_variable_t<i_t> pseudocost_diving(pseudo_costs_t<i_t, f_t>& pc,
                                         const std::vector<i_t>& fractional,
                                         const std::vector<f_t>& solution,
                                         const std::vector<f_t>& root_solution,
                                         logger_t& log);

template <typename i_t, typename f_t>
branch_variable_t<i_t> guided_diving(pseudo_costs_t<i_t, f_t>& pc,
                                     const std::vector<i_t>& fractional,
                                     const std::vector<f_t>& solution,
                                     const std::vector<f_t>& incumbent,
                                     logger_t& log);

// Calculate the variable locks assuming that the constraints
// has the following format: `Ax = b`.
template <typename i_t, typename f_t>
void calculate_variable_locks(const lp_problem_t<i_t, f_t>& lp_problem,
                              std::vector<i_t>& up_locks,
                              std::vector<i_t>& down_locks);

template <typename i_t, typename f_t>
branch_variable_t<i_t> coefficient_diving(const lp_problem_t<i_t, f_t>& lp_problem,
                                          const std::vector<i_t>& fractional,
                                          const std::vector<f_t>& solution,
                                          const std::vector<i_t>& up_locks,
                                          const std::vector<i_t>& down_locks,
                                          logger_t& log);

}  // namespace cuopt::linear_programming::dual_simplex
