/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <mip/problem/problem.cuh>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
int optimal_batch_size_handler(const problem_t<i_t, f_t>& op_problem, int max_batch_size);
}