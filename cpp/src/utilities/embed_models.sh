#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <output_header> <model1.ubj.gz> [model2.ubj.gz ...]"
    exit 1
fi

OUTPUT_HEADER="$1"
shift

# Initialize the header file
echo "// Auto-generated model arrays" > "$OUTPUT_HEADER"
echo "" >> "$OUTPUT_HEADER"

# Collect model names for the mapping array
MODEL_NAMES=""

# Process each model file
for MODEL_FILE in "$@"; do
    # Extract base name without .ubj.gz extension
    MODEL_BASENAME=$(basename "$MODEL_FILE" .ubj.gz)

    echo "Processing: $MODEL_BASENAME from $MODEL_FILE"

    # Create temporary file with proper name so xxd uses it
    TEMP_DIR=$(mktemp -d)
    TEMP_FILE="$TEMP_DIR/${MODEL_BASENAME}.ubj"

    # Decompress the model
    gzip -cd "$MODEL_FILE" > "$TEMP_FILE"

    # Convert to C array with xxd and append to header
    # xxd generates variable names like: _tmp_tmp_xyz_model_ubj
    # We need to replace the entire variable name, not just add to it
    # Add const to place arrays in .rodata section
    xxd -i "$TEMP_FILE" | \
        sed "s/unsigned char [_a-zA-Z0-9]*/const unsigned char ${MODEL_BASENAME}_ubj/" | \
        sed "s/unsigned int [_a-zA-Z0-9]*/const unsigned int ${MODEL_BASENAME}_ubj_len/" >> "$OUTPUT_HEADER"

    # Clean up
    rm -rf "$TEMP_DIR"

    # Add to list for mapping array
    if [ -z "$MODEL_NAMES" ]; then
        MODEL_NAMES="$MODEL_BASENAME"
    else
        MODEL_NAMES="$MODEL_NAMES $MODEL_BASENAME"
    fi
done

# Add model mapping structure
echo "" >> "$OUTPUT_HEADER"
echo "// Model mapping structure" >> "$OUTPUT_HEADER"
echo "struct xgboost_model_info {" >> "$OUTPUT_HEADER"
echo "    const char* name;" >> "$OUTPUT_HEADER"
echo "    const unsigned char* data;" >> "$OUTPUT_HEADER"
echo "    unsigned int length;" >> "$OUTPUT_HEADER"
echo "};" >> "$OUTPUT_HEADER"
echo "" >> "$OUTPUT_HEADER"

# Generate the mapping array
echo "// Array of all available models" >> "$OUTPUT_HEADER"
echo "static const struct xgboost_model_info xgboost_models[] = {" >> "$OUTPUT_HEADER"

for MODEL_NAME in $MODEL_NAMES; do
    echo "    {\"$MODEL_NAME\", ${MODEL_NAME}_ubj, ${MODEL_NAME}_ubj_len}," >> "$OUTPUT_HEADER"
done

echo "};" >> "$OUTPUT_HEADER"
echo "" >> "$OUTPUT_HEADER"
echo "static const unsigned int xgboost_models_count = sizeof(xgboost_models) / sizeof(xgboost_models[0]);" >> "$OUTPUT_HEADER"

echo "Successfully embedded models into $OUTPUT_HEADER"
