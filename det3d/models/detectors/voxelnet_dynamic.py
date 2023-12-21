from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from torch.cuda.amp import autocast as autocast


@DETECTORS.register_module
class VoxelNet_dynamic(SingleStageDetector):
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
        super(VoxelNet_dynamic, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        
        
    def extract_feat(self, example):
        if 'voxels' not in example:
            output = self.reader(example['points'])    
            voxels, coors, shape = output 
        
        x, _ = self.backbone(voxels,coors, len(example['points']), shape)
            
        if self.with_neck:
            x = self.neck(x, example)
        
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
