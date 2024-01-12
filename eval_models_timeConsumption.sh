#!/bin/bash

# Define constants
PROFILE_DELAY=6
EVAL_MODE="time"
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

    echo "$model_type" "$additional_args"
    nsys profile --delay=$PROFILE_DELAY -t nvtx --force-overwrite=true --stats=true \
        --output=$RESULT_DIR/$output_dir_suffix \
        python tools/eval_models.py --eval_mode=$EVAL_MODE --dataset=$DATASET \
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
exit 0

# nusc baseline
nsys profile --delay=6 -t nvtx --force-overwrite=true --stats=true --output=analysis/results/time/nusc_baseline \
python tools/eval_models.py --eval_mode=time --dataset=nuscenes --model_type=baseline

# nusc poolformer
nsys profile --delay=6 -t nvtx --force-overwrite=true --stats=true --output=analysis/results/time/nusc_poolformer \
python tools/eval_models.py --eval_mode=time --dataset=nuscenes --model_type=poolformer

# waymo baseline
nsys profile --delay=6 -t nvtx --force-overwrite=true --stats=true --output=analysis/results/time/waymo_baseline \
python tools/eval_models.py --eval_mode=time --dataset=waymo --model_type=baseline

# waymo poolformer
nsys profile --delay=6 -t nvtx --force-overwrite=true --stats=true --output=analysis/results/time/waymo_poolformer \
python tools/eval_models.py --eval_mode=time --dataset=waymo --model_type=poolformer

