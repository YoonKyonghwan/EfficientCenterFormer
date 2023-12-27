import os
import subprocess
import torch
from det3d.torchie import Config

from det3d.models import build_detector
from det3d.torchie.trainer import load_checkpoint
from torch import nn
import argparse
import onnx
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="train config file path", default="configs/nusc/nuscenes_centerformer_poolformer.py")
    parser.add_argument("--checkpoint", help="the dir to checkpoint which the model read from", default="work_dirs/nuscenes_poolformer/poolformer.pth")
    parser.add_argument("--onnx_dir", help="the dir to save the onnx", default="work_dirs/partition/onnx")
    parser.add_argument("--onnx_name", help="the name of onnx", default="findCenter")
    parser.add_argument("--sanitize", action='store_true', help="whether to sanitize the onnx model")
    args = parser.parse_args()
    return args

    
class CenterFinder(nn.Module):
    def __init__(self, model):
        super(CenterFinder, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model.neck.find_centers(x) 
    

def main():
    args = parse_args()
    config = args.config 
    checkpoint_path = args.checkpoint 

    cfg = Config.fromfile(config)

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, checkpoint_path, map_location="cpu")

    centerFinder = CenterFinder(model)
    centerFinder.eval()

    dummy_input = torch.as_tensor(np.ones([1, 256, 180, 180]), dtype=torch.float32)
        
    onnx_path = os.path.join(args.onnx_dir, args.onnx_name + ".onnx")
    torch.onnx.export(centerFinder, (dummy_input), onnx_path,
                input_names=['input_tensor'], 
                output_names=['ct_feat', 'center_pos_embedding', 'out_scores', 'out_labels', 'out_orders', 'out_masks'],
                export_params=True, 
                do_constant_folding=True,
                opset_version=17,
                # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN,
                verbose=True,
                )
    onnx.checker.check_model(onnx.load(onnx_path))
    
    sanitized_model_name = onnx_path.replace(".onnx", "_sanitized.onnx")
    if args.sanitize:
        # run command in terminal
        subprocess.run(["polygraphy", "surgeon", "sanitize", onnx_path, "--fold-constants", "--output", sanitized_model_name])
    onnx.checker.check_model(onnx.load(sanitized_model_name))
    
    print("success to generate the onnx for findCenter!")
        
if __name__ == "__main__":
    main()