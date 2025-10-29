# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Check for required tools
find_program(XXD_EXECUTABLE xxd)
find_program(GZIP_EXECUTABLE gzip)
find_program(MKTEMP_EXECUTABLE mktemp)

if(NOT XXD_EXECUTABLE)
    message(FATAL_ERROR "xxd command not found. Please install xxd (typically provided by vim package).")
endif()

if(NOT GZIP_EXECUTABLE)
    message(FATAL_ERROR "gzip command not found. Please install gzip.")
endif()

if(NOT MKTEMP_EXECUTABLE)
    message(FATAL_ERROR "mktemp command not found. Please install coreutils or equivalent package.")
endif()

# Directory containing model files
set(XGB_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/utilities/models")
# Find all .ubj.gz model files in models/
file(GLOB XGB_MODEL_FILES "${XGB_MODEL_DIR}/*.ubj.gz")
message(STATUS "XGB_MODEL_DIR: ${XGB_MODEL_DIR}")
message(STATUS "Found ${XGB_MODEL_FILES} model files")

# Set output header name
set(XGB_MODELS_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/utilities/models_ubj.h")

# Path to the embedding script
set(EMBED_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/utilities/embed_models.sh")

# Custom command to generate header by calling the shell script
add_custom_command(
    OUTPUT ${XGB_MODELS_HEADER}
    COMMAND sh ${EMBED_SCRIPT} ${XGB_MODELS_HEADER} ${XGB_MODEL_FILES}
    DEPENDS ${XGB_MODEL_FILES} ${EMBED_SCRIPT}
    COMMENT "Decompressing and embedding all UBJ models into ${XGB_MODELS_HEADER}"
)

add_custom_target(embed_xgboost_models ALL DEPENDS ${XGB_MODELS_HEADER})

set(UTIL_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/utilities/seed_generator.cu
                   ${CMAKE_CURRENT_SOURCE_DIR}/utilities/logger_helper.cpp
                   ${CMAKE_CURRENT_SOURCE_DIR}/utilities/version_info.cpp
                   ${CMAKE_CURRENT_SOURCE_DIR}/utilities/timestamp_utils.cpp
                   ${CMAKE_CURRENT_SOURCE_DIR}/utilities/work_unit_predictor.cpp)

add_subdirectory(linear_programming)
add_subdirectory(math_optimization)
add_subdirectory(mip)

# Only build routing for full builds, not LP-only builds
if(NOT SKIP_ROUTING_BUILD)
  add_subdirectory(routing)
endif()

add_subdirectory(dual_simplex)

set(CUOPT_SRC_FILES ${CUOPT_SRC_FILES} ${UTIL_SRC_FILES} PARENT_SCOPE)
