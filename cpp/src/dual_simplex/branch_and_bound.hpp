/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/bb_event.hpp>
#include <dual_simplex/bnb_worker.hpp>
#include <dual_simplex/deterministic_workers.hpp>
#include <dual_simplex/diving_heuristics.hpp>
#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/mip_node.hpp>
#include <dual_simplex/node_queue.hpp>
#include <dual_simplex/phase2.hpp>
#include <dual_simplex/pseudo_costs.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solution.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/types.hpp>
#include <utilities/macros.cuh>
#include <utilities/omp_helpers.hpp>
#include <utilities/producer_sync.hpp>
#include <utilities/work_limit_timer.hpp>
#include <utilities/work_unit_scheduler.hpp>

#include <omp.h>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

enum class mip_status_t {
  OPTIMAL    = 0,  // The optimal integer solution was found
  UNBOUNDED  = 1,  // The problem is unbounded
  INFEASIBLE = 2,  // The problem is infeasible
  TIME_LIMIT = 3,  // The solver reached a time limit
  NODE_LIMIT = 4,  // The maximum number of nodes was reached (not implemented)
  NUMERICAL  = 5,  // The solver encountered a numerical error
  UNSET      = 6,  // The status is not set
  WORK_LIMIT = 7,  // The solver reached a deterministic work limit
};

enum class mip_solve_mode_t {
  BNB_PARALLEL        = 0,  // Parallel B&B (default)
  BNB_SINGLE_THREADED = 1,  // Single threaded B&B for SubMIP and RINS
};

enum class mip_exploration_status_t {
  UNSET      = 0,  // The status is not set
  TIME_LIMIT = 1,  // The solver reached a time limit
  NODE_LIMIT = 2,  // The maximum number of nodes was reached (not implemented)
  NUMERICAL  = 3,  // The solver encountered a numerical error
  RUNNING    = 4,  // The solver is currently exploring the tree
  COMPLETED  = 5,  // The solver finished exploring the tree
  WORK_LIMIT = 6,  // The solver reached a deterministic work limit
};

enum class node_solve_info_t {
  NO_CHILDREN      = 0,  // The node does not produced children
  UP_CHILD_FIRST   = 1,  // The up child should be explored first
  DOWN_CHILD_FIRST = 2,  // The down child should be explored first
  TIME_LIMIT       = 3,  // The solver reached a time limit
  ITERATION_LIMIT  = 4,  // The solver reached a iteration limit
  NUMERICAL        = 5,  // The solver encounter a numerical error when solving the node
  WORK_LIMIT       = 6,  // The solver reached a deterministic work limit
};

template <typename i_t, typename f_t>
class bounds_strengthening_t;

template <typename i_t, typename f_t>
void upper_bound_callback(f_t upper_bound);

template <typename i_t, typename f_t>
struct opportunistic_tree_update_policy_t;
template <typename i_t, typename f_t>
struct bsp_tree_update_policy_t;

template <typename i_t, typename f_t>
class branch_and_bound_t {
 public:
  branch_and_bound_t(const user_problem_t<i_t, f_t>& user_problem,
                     const simplex_solver_settings_t<i_t, f_t>& solver_settings);

  // Set an initial guess based on the user_problem. This should be called before solve.
  void set_initial_guess(const std::vector<f_t>& user_guess) { guess_ = user_guess; }

  // Set the root solution found by PDLP
  void set_root_relaxation_solution(const std::vector<f_t>& primal,
                                    const std::vector<f_t>& dual,
                                    const std::vector<f_t>& reduced_costs,
                                    f_t objective,
                                    f_t user_objective,
                                    i_t iterations)
  {
    if (!is_root_solution_set) {
      root_crossover_soln_.x              = primal;
      root_crossover_soln_.y              = dual;
      root_crossover_soln_.z              = reduced_costs;
      root_objective_                     = objective;
      root_crossover_soln_.objective      = objective;
      root_crossover_soln_.user_objective = user_objective;
      root_crossover_soln_.iterations     = iterations;
      root_crossover_solution_set_.store(true, std::memory_order_release);
    }
  }

  // Set a solution based on the user problem during the course of the solve
  void set_new_solution(const std::vector<f_t>& solution);

  // BSP-aware solution injection for deterministic mode
  // This queues the solution to be processed at the correct work unit timestamp
  void queue_external_solution_deterministic(const std::vector<f_t>& solution, double work_unit_ts);

  void set_concurrent_lp_root_solve(bool enable) { enable_concurrent_lp_root_solve_ = enable; }

  // Repair a low-quality solution from the heuristics.
  bool repair_solution(const std::vector<f_t>& leaf_edge_norms,
                       const std::vector<f_t>& potential_solution,
                       f_t& repaired_obj,
                       std::vector<f_t>& repaired_solution) const;

  f_t get_lower_bound();
  bool enable_concurrent_lp_root_solve() const { return enable_concurrent_lp_root_solve_; }
  std::atomic<int>* get_root_concurrent_halt() { return &root_concurrent_halt_; }
  void set_root_concurrent_halt(int value) { root_concurrent_halt_ = value; }
  lp_status_t solve_root_relaxation(simplex_solver_settings_t<i_t, f_t> const& lp_settings);

  // The main entry routine. Returns the solver status and populates solution with the incumbent.
  mip_status_t solve(mip_solution_t<i_t, f_t>& solution, mip_solve_mode_t solve_mode);

  work_limit_context_t& get_work_unit_context() { return work_unit_context_; }

  // Get producer sync for external heuristics (e.g., CPUFJ) to register
  producer_sync_t& get_producer_sync() { return producer_sync_; }

 private:
  const user_problem_t<i_t, f_t>& original_problem_;
  const simplex_solver_settings_t<i_t, f_t> settings_;

  work_limit_context_t work_unit_context_{"B&B"};

  // Initial guess.
  std::vector<f_t> guess_;

  // LP relaxation
  csr_matrix_t<i_t, f_t> Arow_;
  lp_problem_t<i_t, f_t> original_lp_;
  std::vector<i_t> new_slacks_;
  std::vector<variable_type_t> var_types_;

  // Variable locks (see definition 3.3 from T. Achterberg, “Constraint Integer Programming,”
  // PhD, Technischen Universität Berlin, Berlin, 2007. doi: 10.14279/depositonce-1634).
  // Here we assume that the constraints are in the form `Ax = b, l <= x <= u`.
  std::vector<i_t> var_up_locks_;
  std::vector<i_t> var_down_locks_;

  // Mutex for upper bound
  omp_mutex_t mutex_upper_;

  // Global variable for upper bound
  omp_atomic_t<f_t> upper_bound_;

  // Global variable for incumbent. The incumbent should be updated with the upper bound
  mip_solution_t<i_t, f_t> incumbent_;

  // Structure with the general info of the solver.
  bnb_stats_t<i_t, f_t> exploration_stats_;

  // Mutex for repair
  omp_mutex_t mutex_repair_;
  std::vector<std::vector<f_t>> repair_queue_;

  // Variables for the root node in the search tree.
  std::vector<variable_status_t> root_vstatus_;
  std::vector<variable_status_t> crossover_vstatus_;
  f_t root_objective_;
  lp_solution_t<i_t, f_t> root_relax_soln_;
  lp_solution_t<i_t, f_t> root_crossover_soln_;
  std::vector<f_t> edge_norms_;
  std::atomic<bool> root_crossover_solution_set_{false};
  bool enable_concurrent_lp_root_solve_{false};
  std::atomic<int> root_concurrent_halt_{0};
  bool is_root_solution_set{false};

  // Pseudocosts
  pseudo_costs_t<i_t, f_t> pc_;

  // Heap storing the nodes waiting to be explored.
  node_queue_t<i_t, f_t> node_queue_;

  // Search tree
  search_tree_t<i_t, f_t> search_tree_;

  // Count the number of workers per type that either are being executed or
  // are waiting to be executed.
  std::array<omp_atomic_t<i_t>, bnb_num_worker_types> active_workers_per_type_;

  // Worker pool
  bnb_worker_pool_t<i_t, f_t> worker_pool_;

  // Global status of the solver.
  omp_atomic_t<mip_status_t> solver_status_;
  omp_atomic_t<bool> is_running_{false};

  // Minimum number of node in the queue. When the queue size is less than
  // this variable, the nodes are added directly to the queue instead of
  // the local stack. This also determines the end of the ramp-up phase.
  i_t min_node_queue_size_;

  // In case, a best-first thread encounters a numerical issue when solving a node,
  // its blocks the progression of the lower bound.
  omp_atomic_t<f_t> lower_bound_ceiling_;

  void report_heuristic(f_t obj);
  void report(char symbol, f_t obj, f_t lower_bound, i_t node_depth);

  // Set the final solution.
  void set_final_solution(mip_solution_t<i_t, f_t>& solution, f_t lower_bound);

  // Update the incumbent solution with the new feasible solution
  // found during branch and bound.
  void add_feasible_solution(f_t leaf_objective,
                             const std::vector<f_t>& leaf_solution,
                             i_t leaf_depth,
                             bnb_worker_type_t thread_type);

  // Repairs low-quality solutions from the heuristics, if it is applicable.
  void repair_heuristic_solutions();

  // We use best-first to pick the `start_node` and then perform a depth-first search
  // from this node (i.e., a plunge). It can only backtrack to a sibling node.
  // Unexplored nodes in the subtree are inserted back into the global heap.
  void plunge_with(bnb_worker_data_t<i_t, f_t>* worker_data, mip_solve_mode_t mode);

  // Perform a deep dive in the subtree determined by the `start_node` in order
  // to find integer feasible solutions.
  void dive_with(bnb_worker_data_t<i_t, f_t>* worker_data);

  // Run the scheduler (aka the master) whose will schedule and manage
  // all the other workers.
  void run_scheduler();

  // Run the branch-and-bound algorithm in single threaded mode.
  // This disable all diving heuristics.
  void single_threaded_solve();

  // Solve the LP relaxation of a leaf node
  dual::status_t solve_node_lp(mip_node_t<i_t, f_t>* node_ptr,
                               bnb_worker_data_t<i_t, f_t>* worker_data,
                               bnb_stats_t<i_t, f_t>& stats,
                               logger_t& log);

  // Selects the variable to branch on.
  branch_variable_t<i_t> variable_selection(mip_node_t<i_t, f_t>* node_ptr,
                                            const std::vector<i_t>& fractional,
                                            bnb_worker_data_t<i_t, f_t>* worker_data);

  // ============================================================================
  // BSP (Bulk Synchronous Parallel) methods for deterministic parallel B&B
  // ============================================================================

  // Main BSP coordinator loop - runs in deterministic mode
  void run_bsp_coordinator(const csr_matrix_t<i_t, f_t>& Arow);

  // Gather all events generated, sort by WU timestamp, apply
  void sort_replay_events(const bb_event_batch_t<i_t, f_t>& events);

  // Prune nodes held by workers based on new incumbent
  void prune_worker_nodes_vs_incumbent();

  // Balance worker loads - redistribute nodes only if significant imbalance detected
  void balance_worker_loads();

  // BSP-specific node solving that records events
  node_solve_info_t solve_node_bsp(bsp_bfs_worker_t<i_t, f_t>& worker,
                                   mip_node_t<i_t, f_t>* node_ptr,
                                   search_tree_t<i_t, f_t>& search_tree,
                                   double current_horizon);

  // Compute accurate lower bound from all BSP sources (called during sync phase)
  f_t compute_bsp_lower_bound();

  // BSP worker loop - runs continuously until scheduler signals stop
  void run_worker_loop(bsp_bfs_worker_t<i_t, f_t>& worker, search_tree_t<i_t, f_t>& search_tree);

  // BSP sync callback - executed when all workers reach barrier
  // Sets bsp_global_termination_status_ if termination is needed
  void bsp_sync_callback(int worker_id);

  // ============================================================================
  // BSP Diving methods
  // ============================================================================

  // Run diving worker loop
  void run_diving_worker_loop(bsp_diving_worker_t<i_t, f_t>& worker);

  // Perform a deterministic dive from the given starting node
  void dive_from_bsp(bsp_diving_worker_t<i_t, f_t>& worker, mip_node_t<i_t, f_t> starting_node);

  // Populate diving heap from BFS worker backlogs at sync
  void populate_diving_heap();

  // Assign starting nodes to diving workers from diving heap
  void assign_diving_nodes();

  // Collect and merge diving solutions at sync
  void collect_diving_solutions();

  template <typename PoolT, typename WorkerTypeGetter>
  void process_worker_solutions(PoolT& pool, WorkerTypeGetter get_worker_type);

  template <typename PoolT>
  void merge_pseudo_cost_updates(PoolT& pool);

  friend struct opportunistic_tree_update_policy_t<i_t, f_t>;
  friend struct bsp_tree_update_policy_t<i_t, f_t>;

 private:
  // BSP state
  // unique_ptr as we only want to initialize these if we're in the determinism codepath
  std::unique_ptr<bsp_bfs_worker_pool_t<i_t, f_t>> bsp_workers_;
  std::unique_ptr<cuopt::work_unit_scheduler_t> bsp_scheduler_;
  mip_status_t bsp_global_termination_status_{mip_status_t::UNSET};
  double bsp_horizon_step_{5.0};     // Work unit step per horizon (tunable)
  double bsp_current_horizon_{0.0};  // Current horizon target
  bool bsp_mode_enabled_{false};     // Whether BSP mode is active
  int bsp_horizon_number_{0};        // Current horizon number (for debugging)

  // Producer synchronization for external heuristics (CPUFJ)
  // B&B waits for registered producers at each horizon sync
  producer_sync_t producer_sync_;

  // Producer wait time statistics
  double total_producer_wait_time_{0.0};
  double max_producer_wait_time_{0.0};
  i_t producer_wait_count_{0};

  // BSP heuristic solution queue - solutions received from GPU heuristics
  // Stored with work unit timestamp for deterministic ordering
  struct queued_heuristic_solution_t {
    std::vector<f_t> solution;
    f_t objective;
    double wut;
  };
  omp_mutex_t mutex_heuristic_queue_;
  std::vector<queued_heuristic_solution_t> heuristic_solution_queue_;

  // ============================================================================
  // BSP Diving state
  // ============================================================================

  // Diving worker pool
  // unique_ptr as we only want to initialize these if we're in the determinism codepath
  std::unique_ptr<bsp_diving_worker_pool_t<i_t, f_t>> bsp_diving_workers_;

  // Diving heap - nodes available for diving, sorted by objective estimate
  struct diving_entry_t {
    mip_node_t<i_t, f_t>* node;
    f_t score;
  };
  struct diving_score_comp {
    bool operator()(const diving_entry_t& a, const diving_entry_t& b) const
    {
      if (a.score != b.score) return a.score > b.score;  // Min-heap by score
      if (a.node->origin_worker_id != b.node->origin_worker_id) {
        return a.node->origin_worker_id > b.node->origin_worker_id;
      }
      return a.node->creation_seq > b.node->creation_seq;
    }
  };
  heap_t<diving_entry_t, diving_score_comp> diving_heap_;
};

}  // namespace cuopt::linear_programming::dual_simplex
