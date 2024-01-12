export CUMM_CUDA_VERSION="11.4"
export CUMM_CUDA_ARCH_LIST="8.7"

# set environment variables
if [ -n "$PYTHONPATH" ]; then
    echo "Warning. PYTHONPATH exist";
fi
export PYTHONPATH=$(pwd):$(pwd)/dependencies/Lidar_AI_Solution/libraries/3DSparseConvolution/tool # for OPT_3D_backbone
export PATH=$PATH:/usr/local/cuda/bin
export CUDA_PATH=/usr/local/cuda
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export LD_PRELOAD=/ssd/jwher96/cf/lib/python3.8/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

