from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from det3d.torchie.trainer import load_checkpoint


@DETECTORS.register_module
class Detector3DTemplate(BaseDetector):
    def __init__(
      self,
      reader,
      backbone,
      map_to_bev,
      neck=None,
      bbox_head=None,
      train_cfg=None,
      test_cfg=None,
      pretrained=None,
    ):
        super().__init__()
        self.reader = builder.build_reader(reader)
        self.backbone = builder.build_backbone(backbone)
        self.map_to_bev = None # TODO make builder
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained is None:
            return 
        try:
            load_checkpoint(self, pretrained, map_location="cpu", strict=False)
            print("init weight from {}".format(pretrained))
        except:
            print("no pretrained model at {}".format(pretrained))
