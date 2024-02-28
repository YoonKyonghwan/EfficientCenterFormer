import itertools
import logging

from types import SimpleNamespace
from det3d.utils.config_tool import get_downsample_factor

tasks = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

# use expanded gt label assigner
window_size = 1

# model settings
model = dict(
    type="DSVT",
    pretrained=None,
    reader=dict(
        type="DynamicPillarVFE",
        model_cfg=SimpleNamespace(
            WITH_DISTANCE=False,
            USE_ABSLOTE_XYZ=True,
            USE_NORM=True,
            NUM_FILTERS=[128, 128],  # channel
            # NUM_FILTERS=[ 192, 192 ],
        ),
        num_point_features=5,
        voxel_size=[0.3, 0.3, 8.0],
        grid_size=[360, 360, 1],
        point_cloud_range=[-54, -54, -5.0, 54, 54, 3.0],
        # voxel_size=[0.23, 0.23, 0.25],
        # grid_size=[468, 468, 32],
        # point_cloud_range=[-53.82, -53.82, -5.0, 53.83, 53.82, 3.0],
    ),
    backbone=dict(
        type="DSVT",
        num_input_features=5,
        ds_factor=8,
        model_cfg=SimpleNamespace(
            INPUT_LAYER=SimpleNamespace(
                sparse_shape=[360, 360, 1],
                downsample_stride=[],
                d_model=[128],  # channel
                set_info=[[90, 4]],
                window_shape=[[30, 30, 1]],
                hybrid_factor=[1, 1, 1],  # x, y, z
                shifts_list=[[[0, 0, 0], [15, 15, 0]]],
                normalize_pos=False,
            ),
            block_name=['DSVTBlock'],
            set_info=[[90, 4]],
            d_model=[128],  # channel
            nhead=[8],
            dim_feedforward=[256],
            dropout=0.0,
            activation="gelu",
            reduction_type='attention',
            conv_out_channel=128,
            output_shape=[360, 360],
        ),
    ),
    map_to_bev=dict(
        type="PointPillarScatter3d",
        model_cfg=SimpleNamespace(
            INPUT_SHAPE=[360, 360, 1],
            NUM_BEV_FEATURES=128,
        ),
        grid_size=[360, 360, 1],
    ),
    neck=dict(
        type="BaseBEVResBackbone",
        model_cfg=SimpleNamespace(
            layer_nums=[1, 2, 2],
            layer_strides=[1, 2, 2],
            num_filters=[128, 128, 256],
            upsample_strides=[0.5, 1, 2],
            num_upsample_filters=[128, 128, 128],
        ),
        input_channels=128,  # num bev features
        # num_input_features=256,
        # tasks=tasks,
        # use_gt_training=True,
        # corner = True,
        # obj_num= 500,
        # assign_label_window_size=window_size,
        # transformer_config=dict(
        #     depth = 2,
        #     MLP_dim = 256,
        #     DP_rate=0.3,
        #     n_points = 15,
        # ),
        # logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="TransFusionHead",
        # class_agnostic=False,
        model_cfg=SimpleNamespace(
            TARGET_ASSIGNER_CONFIG=SimpleNamespace(
                FEATURE_MAP_STRIDE=2,
                DATASET="nuScenes",
                GAUSSIAN_OVERLAP="0.1",
                MIN_RADIUS=2,
                HUNGARIAN_ASSIGNER=dict(
                    cls_cost={'gamma': 2.0, 'alpha': 0.25, 'weight': 0.15},
                    reg_cost={'weight': 0.25},
                    iou_cost={'weight': 0.25},
                ),
            ),
            HIDDEN_CHANNEL=128,
            NUM_PROPOSALS=200,
            BN_MOMENTUM=0.1,
            NMS_KERNEL_SIZE=3,
            NUM_HEADS=8,
            DROPOUT=0,
            ACTIVATION="relu",
            FFN_CHANNEL=256,
            USE_BIAS_BEFORE_NORM=False,
            LOSS_CONFIG=SimpleNamespace(
                LOSS_WEIGHTS={
                    'cls_weight': 1.0,
                    'bbox_weight': 0.25,
                    'hm_weight': 1.0,
                    'loss_iou_rescore_weight': 0.5,
                    'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
                },
                LOSS_CLS=SimpleNamespace(
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                ),
            ),
            SEPARATE_HEAD_CFG=SimpleNamespace(
                HEAD_ORDER=["center", "height", "dim", "rot", "vel"],
                HEAD_DICT={
                    'center': {'out_channels': 2, 'num_conv': 2},
                    'height': {'out_channels': 1, 'num_conv': 2},
                    'dim': {'out_channels': 3, 'num_conv': 2},
                    'rot': {'out_channels': 2, 'num_conv': 2},
                    'vel': {'out_channels': 2, 'num_conv': 2},
                    'iou': {'out_channels': 1, 'num_conv': 2},

                },
            ),
            NUM_HM_CONV=2,
            POST_PROCESSING=SimpleNamespace(
                SCORE_THRESH=0.0,
                POST_CENTER_RANGE=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                USE_IOU_TO_RECTIFY_SCORE=True,
                IOU_RECTIFIER=[0.5],
                NMS_CONFIG=SimpleNamespace(
                    NMS_TYPE="nms_gpu",
                    NMS_THRESH=0.2,
                    NMS_PRE_MAXSIZE=1000,
                    NMS_POST_MAXSIZE=100,
                    SCORE_THRES=0.0,
                )
            ),
        ),
        input_channels=128,         # num bev feature
        num_class=10,
        class_names=class_names,    # dataset.class_names
        grid_size=[360, 360, 1],
        point_cloud_range=[-54, -54, -5.0, 54, 54, 3.0],
        voxel_size=[0.3, 0.3, 8.0],
        predict_boxes_when_training=False,   # ROI head or False
        # in_channels=256,
        # tasks=tasks,
        # dataset='nuscenes',
        # weight=0.25,
        # assign_label_window_size=window_size,
        # corner_loss=True,
        # iou_loss=False,
        # code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
        # common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2), 'vel': (2, 2)},
    ),
)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=2,  # 1440/4 = 360, 360/1 = 360
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    gt_kernel_size=window_size,
    corner_prediction=True,
    pc_range=[-54, -54, -5.0, 54, 54, 3.0],
    voxel_size=[0.3, 0.3, 8.0],
)


train_cfg = dict(assigner=assigner)


test_cfg = dict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=83,
        nms_iou_threshold=0.1,
    ),
    score_threshold=0.03,
    pc_range=[-54, -54],
    out_size_factor=2,      # 360/1 = 360
    voxel_size=[0.3, 0.3],
    obj_num=500,
)


# dataset settings
dataset_type = "NuScenesDataset"
nsweeps = 10
data_root = "data/nuscenes/"

db_sampler = dict(
    type="GT-AUG",
    enable=False,
    db_info_path="data/nuscenes/dbinfos_train_10sweeps_withvelo.pkl",
    sample_groups=[
        dict(car=2),
        dict(truck=3),
        dict(construction_vehicle=7),
        dict(bus=4),
        dict(trailer=6),
        dict(barrier=2),
        dict(motorcycle=6),
        dict(bicycle=6),
        dict(pedestrian=2),
        dict(traffic_cone=2),
    ],
    db_prep_steps=[
        dict(
            filter_by_min_num_points=dict(
                car=5,
                truck=5,
                bus=5,
                trailer=5,
                construction_vehicle=5,
                traffic_cone=5,
                barrier=5,
                motorcycle=5,
                bicycle=5,
                pedestrian=5,
            )
        ),
        dict(filter_by_difficulty=[-1],),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
)

train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    global_rot_noise=[-0.78539816, 0.78539816],
    global_scale_noise=[0.95, 1.05],
    global_translate_noise=0.5,
    db_sampler=db_sampler,
    class_names=class_names,
)
val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
)

voxel_generator = dict(
    range=[-54, -54, -5.0, 54, 54, 3.0],
    voxel_size=[0.3, 0.3, 8.0],
    max_points_in_voxel=10,
    max_voxel_num=[120000, 160000],
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

train_anno = "data/nuscenes/infos_train_10sweeps_withvelo_filter_True.pkl"
val_anno = "data/nuscenes/infos_val_10sweeps_withvelo_filter_True.pkl"
test_anno = None

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        ann_file=val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        test_mode=True,
        ann_file=test_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)


optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.05, fixed_wd=True, moving_average=False,
)  # type="adam_onecycle", lr=0.05, weight_decay=0.05, momentum=0.9
lr_config = dict(
    type="one_cycle", lr_max=0.003, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)  # moms=[0.95, 0.85], pct_start=0.4, div_factor=10, decay_step_list=[35, 45], lr_decay-0.1, lr_clip=0.0000001
# lr_warmup=False, warmup_epoch=1, grad_norm_clip=35, loss_scale_fp16=4.0

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 20
disable_dbsampler_after_epoch = 15
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None
workflow = [('train', 1)]
