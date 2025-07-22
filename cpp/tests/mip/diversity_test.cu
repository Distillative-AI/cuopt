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

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "determinism_utils.cuh"
#include "mip_utils.cuh"

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/utilities/internals.hpp>
#include <linear_programming/initial_scaling_strategy/initial_scaling.cuh>
#include <linear_programming/pdlp.cuh>
#include <linear_programming/restart_strategy/pdlp_restart_strategy.cuh>
#include <linear_programming/step_size_strategy/adaptive_step_size_strategy.hpp>
#include <linear_programming/utilities/problem_checking.cuh>
#include <mip/diversity/diversity_manager.cuh>
#include <mip/feasibility_jump/feasibility_jump.cuh>
#include <mip/local_search/local_search.cuh>
#include <mip/relaxed_lp/relaxed_lp.cuh>
#include <mip/solution/solution.cuh>
#include <mip/solver_context.cuh>
#include <mps_parser/parser.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/seed_generator.cuh>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <thrust/count.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming::test {

void init_handler(const raft::handle_t* handle_ptr)
{
  // Init cuBlas / cuSparse context here to avoid having it during solving time
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    handle_ptr->get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
}

static void setup_device_symbols(rmm::cuda_stream_view stream_view)
{
  raft::common::nvtx::range fun_scope("Setting device symbol");
  detail::set_adaptive_step_size_hyper_parameters(stream_view);
  detail::set_restart_hyper_parameters(stream_view);
  detail::set_pdlp_hyper_parameters(stream_view);
}

static uint32_t test_initial_population_determinism(std::string path,
                                                    unsigned long seed = std::random_device{}())
{
  const raft::handle_t handle_{};

  cuopt::mps_parser::mps_data_model_t<int, double> mps_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, mps_problem);
  problem_checking_t<int, double>::check_problem_representation(op_problem);

  init_handler(op_problem.get_handle_ptr());
  // run the problem constructor of MIP, so that we do bounds standardization
  detail::problem_t<int, double> problem(op_problem);
  problem.preprocess_problem();

  setup_device_symbols(op_problem.get_handle_ptr()->get_stream());

  detail::pdhg_solver_t<int, double> pdhg_solver(problem.handle_ptr, problem);
  detail::pdlp_initial_scaling_strategy_t<int, double> scaling(&handle_,
                                                               problem,
                                                               10,
                                                               1.0,
                                                               pdhg_solver,
                                                               problem.reverse_coefficients,
                                                               problem.reverse_offsets,
                                                               problem.reverse_constraints,
                                                               true);

  auto settings       = mip_solver_settings_t<int, double>{};
  settings.time_limit = 30.;
  auto timer          = cuopt::timer_t(30);
  detail::mip_solver_t<int, double> solver(problem, settings, scaling, timer);
  problem.tolerances = settings.get_tolerances();

  detail::diversity_manager_t<int, double> diversity_manager(solver.context);
  diversity_manager.settings.fj_only_run = true;
  // run with FJ only on the initial population, no recombining
  diversity_manager.run_solver();

  std::vector<uint32_t> hashes;

  auto pop = diversity_manager.get_population_pointer();
  for (const auto& sol : pop->population_to_vector()) {
    hashes.push_back(sol.get_hash());
  }

  return detail::compute_hash(hashes);
}

// TEST(presolve, probing_cache_deterministic)
// {
//   spin_stream_raii_t spin_stream_1;

//   std::vector<std::string> test_instances = {"mip/50v-10-free-bound.mps",
//                                              "mip/neos5-free-bound.mps",
//                                              "mip/neos5.mps",
//                                              "mip/50v-10.mps",
//                                              "mip/gen-ip054.mps",
//                                              "mip/rmatr200-p5.mps"};
//   for (const auto& test_instance : test_instances) {
//     std::cout << "Running: " << test_instance << std::endl;
//     unsigned long seed = std::random_device{}();
//     std::cerr << "Tested with seed " << seed << "\n";
//     auto path          = make_path_absolute(test_instance);
//     uint32_t gold_hash = 0;
//     for (int i = 0; i < 10; ++i) {
//       auto hash = test_probing_cache_determinism(path, seed);
//       if (i == 0) {
//         gold_hash = hash;
//         std::cout << "Gold hash: " << gold_hash << std::endl;
//       } else {
//         EXPECT_EQ(hash, gold_hash);
//       }
//     }
//   }
// }

class DiversityTestParams : public testing::TestWithParam<std::tuple<std::string>> {};

TEST_P(DiversityTestParams, initial_population_deterministic)
{
  cuopt::default_logger().set_pattern("[%n] [%-6l] %v");

  spin_stream_raii_t spin_stream_1;
  spin_stream_raii_t spin_stream_2;

  auto test_instance = std::get<0>(GetParam());
  std::cout << "Running: " << test_instance << std::endl;
  int seed =
    std::getenv("CUOPT_SEED") ? std::stoi(std::getenv("CUOPT_SEED")) : std::random_device{}();
  cuopt::seed_generator::set_seed(seed);
  std::cerr << "Tested with seed " << seed << "\n";
  auto path          = make_path_absolute(test_instance);
  uint32_t gold_hash = 0;
  for (int i = 0; i < 2; ++i) {
    std::cout << "Running " << test_instance << " " << i << std::endl;
    auto hash = test_initial_population_determinism(path, seed);
    if (i == 0) {
      gold_hash = hash;
      std::cout << "Gold hash: " << gold_hash << std::endl;
    } else {
      ASSERT_EQ(hash, gold_hash);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(DiversityTest,
                         DiversityTestParams,
                         testing::Values(
                           // std::make_tuple("mip/sct2.mps"),
                           //               std::make_tuple("mip/thor50dday.mps"),
                           //               std::make_tuple("mip/uccase9.mps"),
                           //               std::make_tuple("mip/neos5-free-bound.mps"),
                           //               std::make_tuple("mip/neos5.mps"),
                           //               std::make_tuple("mip/50v-10.mps"),
                           //               std::make_tuple("mip/rmatr200-p5.mps")
                           std::make_tuple("mip/gen-ip054.mps")));

}  // namespace cuopt::linear_programming::test
