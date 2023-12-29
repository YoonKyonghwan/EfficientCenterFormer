## Getting Started with CenterFormer on nuScenes
This guide is based on modifications from [CenterFormer](https://github.com/TuSimple/centerformer).

### Data Preparation

#### Download and Organize the nuScenes Dataset

Organize the downloaded nuScenes dataset as follows:

```
# For nuScenes Dataset         
└── NUSCENES_DATASET_ROOT
       ├── samples       <-- key frames
       ├── sweeps        <-- frames without annotation
       ├── maps          <-- unused
       ├── v1.0-trainval <-- metadata
```

Create a symbolic link to the dataset root:
```bash
mkdir data && cd data
ln -s DATA_ROOT 
mv DATA_ROOT nuscenes
```
Replace DATA_ROOT with the actual dataset path on your system.


#### Data Creation

Data creation should be under the gpu environment.

```bash
# nuScenes
python tools/create_data.py nuscenes_data_prep \
--root_path=data/nuscenes/ \
--version="v1.0-trainval" --nsweeps=10

mv \
data/nuscenes/infos_val_10sweeps_withvelo_filter_True.pkl \ 
data/nuscenes/infos_val_accuracy.pkl

python tools/sample_testDataset.py
```

In the end, the data and info files should be organized as follows

```
# For nuScenes Dataset 
└── centerformer
       └── data    
              └── nuscenes 
                     ├── samples       <-- key frames
                     ├── sweeps        <-- frames without annotation
                     ├── maps          <-- unused
                     |── v1.0-trainval <-- metadata and annotations
                     |── infos_train_10sweeps_withvelo_filter_True.pkl <-- train annotations
                     |── infos_val_accuracy.pkl <-- val annotations for analyzing accuracy
                     |── infos_val_tim.pkl <-- val annotations for profiling 
                     |── dbinfos_train_10sweeps_withvelo.pkl <-- GT database info files
                     |── gt_database_10sweeps_withvelo <-- GT database 
```

### Training
Training is conducted on a server equipped with 8 GPUs.
```bash
python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py CONFIG_PATH
```

### Evaluation
Evaluation of the model with various optimization options is performed on an OrinNano.

## Prerequisites
Parsing ONNX and building TensorRT engine:
```bash
python tools/export_findCenter_onnx.py --sanitize
python tools/buildEngine.py
python tools/buildEngine.py --fp16
```

## Time Consumption
1. **Profiling**
```bash
bash eval_models_timeConsumption.sh
```
2. **Analysis**
The analysis code can be found in this [script](analysis/time_analysis.ipynb).

## Accuracy
1. **Generate Prediction**
```bash
bash eval_models_accuracy.sh
```
2. **Analysis**
The analysis code can be found in this [script](analysis/time_analysis.ipynb).

