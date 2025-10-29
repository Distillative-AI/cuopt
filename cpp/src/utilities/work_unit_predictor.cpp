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

#include "work_unit_predictor.hpp"

#include <cassert>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>

#include <xgboost/c_api.h>

#include "models_ubj.h"

#define safe_xgboost(call)                                                              \
  {                                                                                     \
    int err = (call);                                                                   \
    if (err != 0) {                                                                     \
      throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                               ": error in " + #call + ":" + XGBGetLastError());        \
    }                                                                                   \
  }

namespace cuopt {

template <typename i_t>
static inline uint32_t compute_hash(std::vector<i_t> h_contents)
{
  // FNV-1a hash

  uint32_t hash = 2166136261u;  // FNV-1a 32-bit offset basis
  std::vector<uint8_t> byte_contents(h_contents.size() * sizeof(i_t));
  std::memcpy(byte_contents.data(), h_contents.data(), h_contents.size() * sizeof(i_t));
  for (size_t i = 0; i < byte_contents.size(); ++i) {
    hash ^= byte_contents[i];
    hash *= 16777619u;
  }
  return hash;
}

work_unit_predictor_t::work_unit_predictor_t(const std::string& model_name) : model_name(model_name)
{
  BoosterHandle booster_handle;
  int ret = XGBoosterCreate(nullptr, 0, &booster_handle);
  safe_xgboost(ret);
  assert(ret == 0);
  if (ret != 0) return;
  raw_handle = reinterpret_cast<void*>(booster_handle);

  // load the embedded model from the .rodata section
  const unsigned char* model_data = nullptr;
  unsigned int model_len          = 0;
  for (unsigned int i = 0; i < xgboost_models_count; i++) {
    if (strcmp(xgboost_models[i].name, model_name.c_str()) == 0) {
      model_data = xgboost_models[i].data;
      model_len  = xgboost_models[i].length;
      break;
    }
  }

  assert(model_data != nullptr);
  assert(model_len > 0);

  ret = XGBoosterLoadModelFromBuffer(booster_handle, model_data, model_len);
  safe_xgboost(ret);
  assert(ret == 0);
  if (ret != 0) return;

  XGBoosterSetParam(booster_handle, "predictor", "gpu_predictor");

  is_valid = true;
}

work_unit_predictor_t::~work_unit_predictor_t()
{
  if (raw_handle != nullptr) {
    BoosterHandle booster_handle = reinterpret_cast<BoosterHandle>(raw_handle);
    XGBoosterFree(booster_handle);
    raw_handle = nullptr;
  }
}

float work_unit_predictor_t::predict_scalar(const std::vector<float>& features, bool verbose) const
{
  assert(is_valid && raw_handle != nullptr);
  if (!is_valid || raw_handle == nullptr) return std::numeric_limits<float>::signaling_NaN();

  // Check cache first
  uint32_t hash = compute_hash(features);
  auto it       = prediction_cache.find(hash);
  if (it != prediction_cache.end()) { return it->second; }

  // Timer: measure elapsed time for prediction
  auto t_start = std::chrono::high_resolution_clock::now();

  // Create DMatrix from feature vector
  DMatrixHandle dmatrix;
  int ret = XGDMatrixCreateFromMat(features.data(),
                                   1,                                        // nrow
                                   features.size(),                          // ncol
                                   std::numeric_limits<float>::quiet_NaN(),  // missing value
                                   &dmatrix);
  safe_xgboost(ret);

  // Predict from DMatrix
  char const config[] =
    "{\"type\": 0, \"iteration_begin\": 0, "
    "\"iteration_end\": 0, \"strict_shape\": true, \"training\": false}";

  const bst_ulong* out_shape = nullptr;
  bst_ulong out_dim          = 0;
  const float* out_result    = nullptr;
  ret = XGBoosterPredictFromDMatrix(reinterpret_cast<BoosterHandle>(raw_handle),
                                    dmatrix,
                                    config,
                                    &out_shape,
                                    &out_dim,
                                    &out_result);
  safe_xgboost(ret);

  float prediction = out_result[0];

  // Free DMatrix
  XGDMatrixFree(dmatrix);

  auto t_end        = std::chrono::high_resolution_clock::now();
  double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
  printf("[work_unit_predictor_t::predict_scalar] Prediction took %.3f ms\n", elapsed_ms);

  // Store in cache
  prediction_cache[hash] = prediction;

  return prediction;
}

float work_unit_predictor_t::predict_scalar(const std::map<std::string, float>& feature_map,
                                            bool verbose) const
{
  // Extract features in the expected order for the model
  // Order matches training data: [target_time, n_of_minimums_for_exit, n_variables, n_constraints,
  // nnz, sparsity, nnz_stddev, unbalancedness]
  std::vector<float> features;
  features.reserve(feature_map.size());

  // Add features in the order expected by the model
  // This order should match what was used during training
  const std::vector<std::string> feature_order = {"target_time",
                                                  "n_of_minimums_for_exit",
                                                  "n_variables",
                                                  "n_constraints",
                                                  "nnz",
                                                  "sparsity",
                                                  "nnz_stddev",
                                                  "unbalancedness"};

  for (const auto& name : feature_order) {
    auto it = feature_map.find(name);
    if (it != feature_map.end()) {
      features.push_back(it->second);
    } else {
      // Feature not found - use default value of 0
      features.push_back(0.0f);
    }
  }

  return predict_scalar(features, verbose);
}

}  // namespace cuopt
