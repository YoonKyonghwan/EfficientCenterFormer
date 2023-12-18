import torch
from det3d.torchie import Config

import pickle
from det3d.models import build_detector
from det3d.torchie.trainer import load_checkpoint
from torch import nn

config = "/workspace/centerformer/configs/nusc/nuscenes_centerformer_poolformer.py"

cfg = Config.fromfile(config)
FINDCENTER_GEN_ONNX = True


checkpoint_path = "/workspace/centerformer/work_dirs/nuscenes_poolformer/poolformer.pth"

model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu");


class CenterFinder(nn.Module):
    def __init__(self, model):
        super(CenterFinder, self).__init__()
        self.model = model

    def forward(self, x):
        # x = self.model.neck(x)
        # preds = self.model.bbox_head(x)
        # return preds
        return self.model.neck.find_centers(x) # pos_features, out_dict_list, ct_feat 
    
centerFinder = CenterFinder(model)
centerFinder.cuda()
centerFinder.eval();

pickle_dir = "/workspace/centerformer/work_dirs/partition/sample_data/"

with open(pickle_dir + "findcenter_input.pkl", 'rb') as handle:
    x = pickle.load(handle)
    
with torch.no_grad():
    ct_feat, center_pos_embedding, out_scores, out_labels, out_order, out_mask = centerFinder(x)
    
    import onnx

if FINDCENTER_GEN_ONNX:
    model_name = "findCenter.onnx"
    torch.onnx.export(centerFinder, x, model_name, do_constant_folding=True,
                    input_names=['input_tensor'],
                    output_names=['ct_feat', 'center_pos_embedding', 'out_scores', 'out_labels', 'out_order', 'out_mask'],
                    export_params=True, opset_version=11)
    
    # !export POLYGRAPHY_AUTOINSTALL_DEPS=1
    #!polygraphy surgeon sanitize findCenter.onnx --fold-constants --output findCenter_folded.onnx

    onnx.checker.check_model(onnx.load("findCenter_folded.onnx"))
    print("gen findCenter.onnx success!")
else:
    print("pass")