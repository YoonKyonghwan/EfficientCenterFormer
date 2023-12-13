from ..registry import DETECTORS
from .single_stage import SingleStageDetector
import torch 
from torch.cuda.amp import autocast as autocast

import nvtx
import pickle

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
        with nvtx.annotate("reader"):
            if 'voxels' not in example:
                output = self.reader(example['points'])    
                voxels, coors, shape = output 

                # data = dict(
                #     features=voxels,
                #     coors=coors,
                #     batch_size=len(example['points']),
                #     input_shape=shape,
                #     voxels=voxels
                # )
                
        # with open("/workspace/centerformer/work_dirs/partition/sample_data/reader_output.pkl", 'wb') as handle:
        #     pickle.dump(output, handle)
        
        with nvtx.annotate("3D_backbone"):
            # x, voxel_feature = self.backbone(
            #         data['voxels'], data["coors"], data["batch_size"], data["input_shape"]
            #     )
            x, _ = self.backbone(voxels,coors, len(example['points']), shape)
            
        # with open("/workspace/centerformer/work_dirs/partition/sample_data/backbone_output.pkl", 'wb') as handle:
        #     pickle.dump(x, handle)

        if self.with_neck:
            x = self.neck(x, example)
        
        # with open("/workspace/centerformer/work_dirs/partition/sample_data/neck_output.pkl", 'wb') as handle:
        #     pickle.dump(x, handle)
        
        return x

    # def forward(self, example, return_loss=True, **kwargs):
    def forward(self, example, return_loss=False):
        
        # with open("/workspace/centerformer/work_dirs/partition/sample_data/example.pkl", 'wb') as handle:
        #     pickle.dump(example, handle)
            
        x = self.extract_feat(example)
        
        
        with nvtx.annotate("bbox_head"):
            preds = self.bbox_head(x)
            
        # with open("/workspace/centerformer/work_dirs/partition/sample_data/bbox_head_output.pkl", 'wb') as handle:
        #     pickle.dump(preds, handle)
                
        with nvtx.annotate("post_processing"):
            if return_loss:
                return self.bbox_head.loss(example, preds, self.test_cfg)
            else:
                # temp = self.bbox_head.predict(example, preds, self.test_cfg)
                # with open("/workspace/centerformer/work_dirs/partition/sample_data/predict_output.pkl", 'wb') as handle:
                #     pickle.dump(temp, handle)
                return self.bbox_head.predict(example, preds, self.test_cfg)


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
