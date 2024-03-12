import torch
from types import SimpleNamespace

from det3d.models.backbones.map_to_bev.pointpillar3d_scatter import PointPillarScatter3d
from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from torch.cuda.amp import autocast as autocast


@DETECTORS.register_module
class PoolNet(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(PoolNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        # TODO make builder
        self.map_to_bev = PointPillarScatter3d(
            model_cfg=SimpleNamespace(
                INPUT_SHAPE=[360, 360, 1],
                NUM_BEV_FEATURES=128,       # channel
            ),
            grid_size=[360, 360, 1],
        )
        # pool_size = 3
        # exp1
        # self.downsample = torch.nn.AvgPool2d(pool_size, stride=2, padding=pool_size//2, count_include_pad=False)
        # exp2
        # self.downsample = torch.nn.Sequential(
        #     torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
        #     torch.nn.ReLU(),
        # )
        # exp3, exp5
        # self.downsample = torch.nn.Sequential(
        #     torch.nn.Conv2d(in_channels=128, out_channels=8, kernel_size=1, stride=1, bias=False),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(in_channels=8, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
        # )
        # exp4
        # no downsample
        # exp6
        # self.downsample = DownSampleBlock()

        
    def extract_feat(self, example):
        points = []
        for i in range(len(example["points"])):
            point_num = example["points"][i].shape[0]
            device    = example["points"][i].device
            batch_idx = torch.full((point_num, 1), i, dtype=torch.int32).to(device)
            points.append(torch.cat((batch_idx, example["points"][i]), dim=1))
            del batch_idx
        example["points"] = torch.cat(points, dim=0)
        del points

        example = self.reader(example)
        example = self.backbone(example)  # [2, 128, 360, 360]
        example = self.map_to_bev(example)# [2, 128, 360, 360]
        x = example["spatial_features"]   # [2, 128, 360, 360] [batch, NUM_BEV_FEATURES, x, y]
        # x = self.downsample(x)            # [2, 256, 180, 180]
        
        if self.with_neck:
            x = self.neck(x, example)   # [1, 256, 180, 180] [batch, num_input_features, x, y]
        
        return x


    def forward(self, example, return_loss=False):
        x = self.extract_feat(example)
        
        preds = self.bbox_head(x)
        if return_loss:
            return self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return self.bbox_head.predict(example['metadata'], preds, self.test_cfg)
            
            
    def extract_feat_baseline(self, example):
        if 'voxels' not in example:
            output = self.reader(example['points'])    
            voxels, coors, shape = output 

            data = dict(
                features=voxels,
                coors=coors,
                batch_size=len(example['points']),
                input_shape=shape,
                voxels=voxels
            )
            
        x, voxel_feature = self.backbone(
                data['voxels'], data["coors"], data["batch_size"], data["input_shape"]
            )

        if self.with_neck:
            x = self.neck(x, example)

        return x, voxel_feature
    

    def forward_baseline(self, example, return_loss=True, **kwargs):
        x, _ = self.extract_feat_baseline(example)
        preds = self.bbox_head.forward_baseline(x)

        if return_loss:
            return self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return self.bbox_head.predict_baseline(example, preds, self.test_cfg)


    def forward_two_stage(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(example, data)
        bev_feature = x['BEV_feat']
        preds = self.bbox_head(x)

        # manual deepcopy ...
        new_preds = []
        for pred in preds:
            new_pred = {} 
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        if return_loss:
            return boxes, bev_feature, self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return boxes, bev_feature, None 
