/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

class pdlp_predictor {
 public:
  union Entry {
    int missing;
    float fvalue;
    int qvalue;
  };

  static int32_t get_num_target(void);
  static void get_num_class(int32_t* out);
  static int32_t get_num_feature(void);
  static const char* get_threshold_type(void);
  static const char* get_leaf_output_type(void);
  static void predict(union Entry* data, int pred_margin, double* result);
  static void postprocess(double* result);
  static int quantize(float val, unsigned fid);

  // Feature names
  static constexpr int NUM_FEATURES = 8;
  static const char* feature_names[NUM_FEATURES];
};  // class pdlp_predictor
