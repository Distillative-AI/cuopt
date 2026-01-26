/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#pragma once

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

class fj_predictor {
 public:
  union Entry {
    int missing;
    double fvalue;
    int qvalue;
  };

  static int32_t get_num_target(void);
  static void get_num_class(int32_t* out);
  static int32_t get_num_feature(void);
  static const char* get_threshold_type(void);
  static const char* get_leaf_output_type(void);
  static void predict(Entry* data, int pred_margin, double* result);
  static void postprocess(double* result);
  static int quantize(double val, unsigned fid);

  // Feature names
  static constexpr int NUM_FEATURES = 12;
  static const char* feature_names[NUM_FEATURES];
};  // class fj_predictor
