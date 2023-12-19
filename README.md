# Optimization of Inference with CenterFormer

## Highlights
- **ToDO** need to describe the optimization technique that we use


## Installation
Please refer to [INSTALL](docs/INSTALL.md) to set up libraries needed for distributed training and sparse convolution.

## Training and Evaluation
Please refer to [WAYMO](docs/WAYMO.md) and [nuScenes](docs/NUSC.md) to prepare the data, training and evaluation.

## Result

#### 3D detection on Waymo test set 

|         |  #Frame | Veh_L2 | Ped_L2 | Cyc_L2  | Mean   |
|---------|---------|--------|--------|---------|---------|
| CenterFormer| 8       |   77.7     |  76.6      |   72.4      |  75.6    |
| CenterFormer| 16      |   78.3     |  77.4      |   73.2      |  76.3    |

#### 3D detection on Waymo val set 

|         |  #Frame | Veh_L2 | Ped_L2 | Cyc_L2  | Mean   |
|---------|---------|--------|--------|---------|---------|
| [CenterFormer](configs/waymo/voxelnet/waymo_centerformer.py)| 1       |   69.4     |  67.7      |   70.2      |  69.1    |
| [CenterFormer deformable](configs/waymo/voxelnet/waymo_centerformer_deformable.py)| 1       |   69.7     |  68.3      |   68.8      |  69.0    |
| [CenterFormer](configs/waymo/voxelnet/waymo_centerformer_multiframe_2frames.py)| 2       |   71.7     |  73.0      |   72.7      |  72.5    |
| [CenterFormer deformable](configs/waymo/voxelnet/waymo_centerformer_multiframe_deformable_2frames.py)| 2       |   71.6     |  73.4      |   73.3      |  72.8    |
| [CenterFormer deformable](configs/waymo/voxelnet/waymo_centerformer_multiframe_deformable_4frames.py)| 4       |   72.9     |  74.2      |   72.6      |  73.2    |
| [CenterFormer deformable](configs/waymo/voxelnet/waymo_centerformer_multiframe_deformable_8frames.py)| 8       |   73.8     |  75.0      |   72.3      |  73.7    |
| [CenterFormer deformable](configs/waymo/voxelnet/waymo_centerformer_multiframe_deformable_16frames.py)| 16      |   74.6     |  75.6      |   72.7      |  74.3    |

#### 3D detection on nuScenes val set
|         |  NDS    | mAP    |
|---------|---------|--------|
| [CenterFormer](configs/nusc/nuscenes_centerformer_separate_detection_head.py)| 68.0     |  62.7      |
| [CenterFormer deformable](configs/nusc/nuscenes_centerformer_deformable_separate_detection_head.py)| 68.4     |  63.0      |

The training and evaluation configs of the above models are provided in [Configs](configs/waymo/README.md).


## Acknowlegement
This project is developed based on the [CenterPoint](https://github.com/tianweiy/CenterPoint) codebase. We use the deformable cross-attention implementation from [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR).
