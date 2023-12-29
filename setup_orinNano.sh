# check the number of arguments
if [ $# -ne 1 ]; then
    echo "Please input the path of nuscenes dataset"
    exit 1
fi

echo "This setup.sh works for Jetson Orin Nano with Jetpack 5.1.2"
echo "Please make sure you have installed Jetpack 5.1.2 and CUDA 11.4"
echo "If you have other Hardware and Software, please modify this file accordingly"

# install dependencies
pip install --upgrade pip
pip install --no-cache https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
pip install -r requirements.txt

export CUMM_CUDA_VERSION="11.4"
export CUMM_CUDA_ARCH_LIST="8.7"

mkdir -p data/nuscenes
cd data/nuscenes
ln -s $1

mkdir dependencies
cd dependencies
# intstall cumm & spconv
git clone https://github.com/FindDefinition/cumm
cd cumm/
pip install -e .
cd ..
git clone https://github.com/traveller59/spconv 
cd spconv/
sed -i 's/\, "cumm/\]\#\, "cumm/g' pyproject.toml # remove cumm requirement
pip install -e .
cd ..
# build Optimized_SDsparseconvolution
git clone --recursive https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution
cd Lidar_AI_Solution/libraries/3DSparseConvolution/tool
make pyscn -j8 

# build extensions
cd det3d/ops/iou3d_nms
python setup.py build_ext --inplace

cd ../.. && cd models/ops/
python setup.py build install

# set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/dependencies/Lidar_AI_Solution/libraries/3DSparseConvolution/tool # for OPT_3D_backbone
export PATH=$PATH:/usr/local/cuda/bin
export CUDA_PATH=/usr/local/cuda
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

python tools/export_findCenter_onnx.py --sanitize
python tools/buildEngine.py
python tools/buildEngine.py --fp16