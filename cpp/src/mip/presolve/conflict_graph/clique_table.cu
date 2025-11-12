/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define DEBUG_KNAPSACK_CONSTRAINTS 1

#include "clique_table.cuh"

#include <dual_simplex/sparse_matrix.hpp>
#include <mip/mip_constants.hpp>
#include <utilities/logger.hpp>
#include <utilities/macros.cuh>

namespace cuopt::linear_programming::detail {

// do constraints with only binary variables.
template <typename i_t, typename f_t>
void find_cliques_from_constraint(const knapsack_constraint_t<i_t, f_t>& kc,
                                  clique_table_t<i_t, f_t>& clique_table)
{
  i_t size = kc.entries.size();
  cuopt_assert(size > 1, "Constraint has not enough variables");
  if (kc.entries[size - 1].val + kc.entries[size - 2].val <= kc.rhs) { return; }
  std::vector<i_t> clique;
  i_t k = size - 1;
  // find the first clique, which is the largest
  // FIXME: do binary search
  while (k >= 0) {
    if (kc.entries[k].val + kc.entries[k - 1].val <= kc.rhs) { break; }
    clique.push_back(kc.entries[k].col);
    k--;
  }
  clique_table.first.push_back(clique);
  const i_t original_clique_start_idx = k;
  // find the additional cliques
  k--;
  while (k >= 0) {
    f_t curr_val = kc.entries[k].val;
    i_t curr_col = kc.entries[k].col;
    // do a binary search in the clique coefficients to find f, such that coeff_k + coeff_f > rhs
    // this means that we get a subset of the original clique and extend it with a variable
    f_t val_to_find = kc.rhs - curr_val + 1e-6;
    auto it         = std::lower_bound(
      kc.entries.begin() + original_clique_start_idx, kc.entries.end(), val_to_find);
    if (it != kc.entries.end()) {
      i_t position_on_knapsack_constraint = std::distance(kc.entries.begin(), it);
      i_t start_pos_on_clique = position_on_knapsack_constraint - original_clique_start_idx;
      cuopt_assert(start_pos_on_clique >= 1, "Start position on clique is negative");
      cuopt_assert(it->val + curr_val > kc.rhs, "RHS mismatch");
#if DEBUG_KNAPSACK_CONSTRAINTS
      CUOPT_LOG_DEBUG("Found additional clique: %d, %d, %d",
                      curr_col,
                      clique_table.first.size() - 1,
                      start_pos_on_clique);
#endif
      clique_table.addtl_cliques.push_back(
        {curr_col, (i_t)clique_table.first.size() - 1, start_pos_on_clique});
    } else {
      break;
    }
    k--;
  }
}

// sort CSR by constraint coefficients
template <typename i_t, typename f_t>
void sort_csr_by_constraint_coefficients(
  std::vector<knapsack_constraint_t<i_t, f_t>>& knapsack_constraints)
{
  // sort the rows of the CSR matrix by the coefficients of the constraint
  for (auto& knapsack_constraint : knapsack_constraints) {
    std::sort(knapsack_constraint.entries.begin(), knapsack_constraint.entries.end());
  }
}

template <typename i_t, typename f_t>
void make_coeff_positive_knapsack_constraint(
  const dual_simplex::user_problem_t<i_t, f_t>& problem,
  std::vector<knapsack_constraint_t<i_t, f_t>>& knapsack_constraints)
{
  for (auto& knapsack_constraint : knapsack_constraints) {
    f_t rhs_offset = 0;
    for (auto& entry : knapsack_constraint.entries) {
      if (entry.val < 0) {
        entry.val = -entry.val;
        rhs_offset += entry.val;
        // negation of a variable is var + num_cols
        entry.col = entry.col + problem.num_cols;
      }
    }
    knapsack_constraint.rhs += rhs_offset;
    cuopt_assert(knapsack_constraint.rhs >= 0, "RHS must be non-negative");
  }
}

// convert all the knapsack constraints
// if a binary variable has a negative coefficient, put its negation in the constraint
template <typename i_t, typename f_t>
void fill_knapsack_constraints(const dual_simplex::user_problem_t<i_t, f_t>& problem,
                               std::vector<knapsack_constraint_t<i_t, f_t>>& knapsack_constraints)
{
  dual_simplex::csr_matrix_t<i_t, f_t> A(0, 0, 0);
  problem.A.to_compressed_row(A);
  // we might add additional constraints for the equality constraints
  i_t added_constraints = 0;
  for (i_t i = 0; i < A.m; i++) {
    std::pair<i_t, i_t> constraint_range = A.get_constraint_range(i);
    if (constraint_range.second - constraint_range.first < 2) {
      CUOPT_LOG_DEBUG("Constraint %d has less than 2 variables, skipping", i);
      continue;
    }
    bool all_binary = true;
    // check if all variables are binary
    for (i_t j = constraint_range.first; j < constraint_range.second; j++) {
      if (problem.var_types[A.j[j]] != dual_simplex::variable_type_t::INTEGER ||
          problem.lower[A.j[j]] != 0 || problem.upper[A.j[j]] != 1) {
        all_binary = false;
        break;
      }
    }
    // if all variables are binary, convert the constraint to a knapsack constraint
    if (!all_binary) { continue; }
    knapsack_constraint_t<i_t, f_t> knapsack_constraint;

    knapsack_constraint.cstr_idx = i;
    if (problem.row_sense[i] == 'L') {
      knapsack_constraint.rhs = problem.rhs[i];
      for (i_t j = constraint_range.first; j < constraint_range.second; j++) {
        knapsack_constraint.entries.push_back({A.j[j], A.x[j]});
      }
    } else if (problem.row_sense[i] == 'G') {
      knapsack_constraint.rhs = -problem.rhs[i];
      for (i_t j = constraint_range.first; j < constraint_range.second; j++) {
        knapsack_constraint.entries.push_back({A.j[j], -A.x[j]});
      }
    } else if (problem.row_sense[i] == 'E') {
      // less than part
      knapsack_constraint.rhs = problem.rhs[i];
      for (i_t j = constraint_range.first; j < constraint_range.second; j++) {
        knapsack_constraint.entries.push_back({A.j[j], A.x[j]});
      }
      // greater than part: convert it to less than
      knapsack_constraint_t<i_t, f_t> knapsack_constraint2;
      knapsack_constraint2.cstr_idx = A.m + added_constraints++;
      knapsack_constraint2.rhs      = -problem.rhs[i];
      for (i_t j = constraint_range.first; j < constraint_range.second; j++) {
        knapsack_constraint2.entries.push_back({A.j[j], -A.x[j]});
      }
      knapsack_constraints.push_back(knapsack_constraint2);
    }
    knapsack_constraints.push_back(knapsack_constraint);
  }
  CUOPT_LOG_DEBUG("Number of knapsack constraints: %d added %d constraints",
                  knapsack_constraints.size(),
                  added_constraints);
}

template <typename i_t, typename f_t>
void print_knapsack_constraints(
  const std::vector<knapsack_constraint_t<i_t, f_t>>& knapsack_constraints)
{
#if DEBUG_KNAPSACK_CONSTRAINTS
  std::cout << "Number of knapsack constraints: " << knapsack_constraints.size() << "\n";
  for (const auto& knapsack : knapsack_constraints) {
    std::cout << "Knapsack constraint idx: " << knapsack.cstr_idx << "\n";
    std::cout << "  RHS: " << knapsack.rhs << "\n";
    std::cout << "  Entries:\n";
    for (const auto& entry : knapsack.entries) {
      std::cout << "    col: " << entry.col << ", val: " << entry.val << "\n";
    }
    std::cout << "----------\n";
  }
#endif
}

template <typename i_t, typename f_t>
void print_clique_table(const clique_table_t<i_t, f_t>& clique_table)
{
#if DEBUG_KNAPSACK_CONSTRAINTS
  std::cout << "Number of cliques: " << clique_table.first.size() << "\n";
  for (const auto& clique : clique_table.first) {
    std::cout << "Clique: ";
    for (const auto& var : clique) {
      std::cout << var << " ";
    }
  }
  std::cout << "Number of additional cliques: " << clique_table.addtl_cliques.size() << "\n";
  for (const auto& addtl_clique : clique_table.addtl_cliques) {
    std::cout << "Additional clique: " << addtl_clique.vertex_idx << ", " << addtl_clique.clique_idx
              << ", " << addtl_clique.start_pos_on_clique << "\n";
  }
#endif
}

template <typename i_t, typename f_t>
void find_initial_cliques(const dual_simplex::user_problem_t<i_t, f_t>& problem)
{
  std::vector<knapsack_constraint_t<i_t, f_t>> knapsack_constraints;
  fill_knapsack_constraints(problem, knapsack_constraints);
  make_coeff_positive_knapsack_constraint(problem, knapsack_constraints);
  sort_csr_by_constraint_coefficients(knapsack_constraints);
  // print_knapsack_constraints(knapsack_constraints);
  clique_table_t<i_t, f_t> clique_table;
  for (const auto& knapsack_constraint : knapsack_constraints) {
    find_cliques_from_constraint(knapsack_constraint, clique_table);
  }
  print_clique_table(clique_table);
  exit(0);
}

#define INSTANTIATE(F_TYPE)                        \
  template void find_initial_cliques<int, F_TYPE>( \
    const dual_simplex::user_problem_t<int, F_TYPE>& problem);
#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif
#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif
#undef INSTANTIATE

}  // namespace cuopt::linear_programming::detail
