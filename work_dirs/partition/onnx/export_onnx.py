import onnx
import numpy as np
import torch
from torch import nn
from det3d.torchie import Config

import pickle
from det3d.models import build_detector
from det3d.torchie.trainer import load_checkpoint


config = "/workspace/centerformer/configs/nusc/nuscenes_centerformer_poolformer.py"
checkpoint_path = "/workspace/centerformer/work_dirs/nuscenes_poolformer/poolformer.pth"

cfg = Config.fromfile(config)
model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu");

class CenterFinder(nn.Module):
    def __init__(self, model):
        super(CenterFinder, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model.neck.find_centers(x) 
    
centerFinder = CenterFinder(model)
# centerFinder.cuda()
centerFinder.eval();


model_name = "findCenter.onnx"
# dummy_input=torch.randn(x.shape, requires_grad=True).cuda()
dummy_input = torch.as_tensor(np.ones([1, 256, 180, 180]), dtype=torch.float32)
torch.onnx.export(centerFinder, (dummy_input), model_name,
            input_names=['input_tensor'], 
            output_names=['ct_feat', 'center_pos_embedding', 'out_scores', 'out_labels', 'out_orders', 'out_masks'],
            export_params=True, 
            do_constant_folding=True,
            opset_version=17,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
            verbose=True,
            # training=torch.onnx.TrainingMode.TRAINING,
            )

# !export POLYGRAPHY_AUTOINSTALL_DEPS=1
# !polygraphy surgeon sanitize findCenter.onnx --fold-constants --output findCenter_folded.onnx

onnx.checker.check_model(onnx.load("findCenter_folded.onnx"))
print("gen findCenter.onnx success!")
