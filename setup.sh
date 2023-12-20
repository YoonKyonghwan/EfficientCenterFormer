# install dependencies
cuda_version=$1
case $cuda_version in
    "11.8")
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
        pip install spconv-cu118;;
    "12.0"|"12.1"|"12.2")
        conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
        pip install spconv-cu120;;
    *)
        echo "Unsupported CUDA version: $cuda_version";;
esac

pip install -r requirements.txt

cd det3d/ops/iou3d_nms
python setup.py build_ext --inplace

cd ../.. && cd models/ops/
python setup.py build install
python test.py


tensorRT_version=$2
case $tensorRT_version in
    "8.5")
        pip install tensorrt==8.5.1.7;;
    "8.6")
        pip install tensorrt==8.6.1;;
    *)
        echo "Unsupported tensorRT version: $tensorRT_version";;
esac

# set environment variables
current_path=$(pwd)
export PYTHONPATH=$PYTHONPATH:$current_path
export PATH=$PATH:/usr/local/cuda/bin
export CUDA_PATH=/usr/local/cuda
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/anaconda3/envs/trt/lib