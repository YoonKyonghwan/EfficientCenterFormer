echo "This setup.sh works for Jetson Orin Nano with Jetpack 5.1.2"
echo "Please make sure you have installed Jetpack 5.1.2 and CUDA 11.4"
echo "If you have other Hardware and Software, please modify this file accordingly"

# install dependencies
pip install --upgrade pip
export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
pip install --no-cache $TORCH_INSTALL
pip install -r requirements.txt
pip install spconv-cu114
pip install tensorrt==8.5.1.7;;

# build extensions
cd det3d/ops/iou3d_nms
python setup.py build_ext --inplace

cd ../.. && cd models/ops/
python setup.py build install
# python test.py

# set environment variables
current_path=$(pwd)
export PYTHONPATH=$PYTHONPATH:$current_path
export PATH=$PATH:/usr/local/cuda/bin
export CUDA_PATH=/usr/local/cuda
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/anaconda3/envs/trt/lib