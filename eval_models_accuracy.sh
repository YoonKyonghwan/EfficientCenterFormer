#!/bin/bash

# Define constants
PROFILE_DELAY=125
EVAL_MODE="accuracy"
RESULT_DIR="analysis/results/$EVAL_MODE"
DATASET="nuscenes"

# Function to perform profiling
profile_model() {
    local model_type=$1
    local additional_args=$2
    local output_dir_suffix=$3

    # Create output directory
    if [ ! -d "$RESULT_DIR" ]; then
        mkdir -p $RESULT_DIR
    fi

    echo "$model_type"
    python tools/eval_models.py --eval_mode=$EVAL_MODE --dataset=$DATASET \
        --work_dir=$RESULT_DIR/$output_dir_suffix \
        --model_type=$model_type $additional_args
}

# Profiling different model configurations
profile_model "baseline" "" "baseline"
profile_model "poolformer" "" "poolformer"
profile_model "poolformer" "--mem_opt" "mem_opt"
profile_model "poolformer" "--mem_opt --trt" "trt_32"
profile_model "poolformer" "--mem_opt --trt --trt_fp16" "trt_16"
profile_model "poolformer" "--mem_opt --trt --trt_fp16 --backbone_opt" "backbone_opt"

# Completion message
echo "profiling done"
echo "ready to analyze results by analysis/timeAnalysis.ipynb"
