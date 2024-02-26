from ..registry import DETECTORS
from .detector3d_template import Detector3DTemplate


@DETECTOR.register_module
class DSVT(Detector3DTemplate):
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
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        super(PoolNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        # TODO make builder
        self.map_to_bev = PointPillarScatter3d(
            model_cfg=SimpleNamespace(
                INPUT_SHAPE=[360, 360, 1],
                NUM_BEV_FEATURES=128,
            ),
            grid_size=[360, 360, 1],
        )

    def extract_feat(self, example):
        # convert list([x, y, z, i, e]) to tensor(batch_idx, x, y, z, i, e)
        points = []
        for i in range(len(example["points"])):
            # 5=(x, y, z, i, e).shape[0]
            point_num = example["points"][i].shape[0]
            device = example["points"][i].device     # gpu device
            # make tensor(batch_idx, 1) which value is idx(i)
            batch_idx = torch.full(
                (point_num, 1), i, dtype=torch.int32).to(device)
            points.append(torch.cat((batch_idx, example["points"][i]), dim=1))
            del batch_idx
        example["points"] = torch.cat(points, dim=0)
        del points

        example = self.reader(example)
        example = self.backbone(example)  # [2, 128, 360, 360]
        example = self.map_to_bev(example)  # [2, 128, 360, 360]

        if self.with_neck:
            # [2, 256, 180, 180] [batch, num_input_features, x, y]
            x = self.neck(x, example)

        return x

    def forward(self, batch_dict):
        x = self.extract_feat(example)

        preds = self.bbox_head(x)
        if return_loss:
            return self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return self.bbox_head.predict(example['metadata'], preds, self.test_cfg)

    # def get_training_loss(self, batch_dict):
    #     disp_dict = {}

    #     loss_trans, tb_dict = batch_dict['loss'], batch_dict['tb_dict']
    #     tb_dict = {
    #         'loss_trans': loss_trans.item(),
    #         **tb_dict
    #     }

    #     loss = loss_trans
    #     return loss, tb_dict, disp_dict

    # def post_processing(self, batch_dict):
    #     post_process_cfg = self.model_cfg.POST_PROCESSING
    #     batch_size = batch_dict['batch_size']
    #     final_pred_dict = batch_dict['final_box_dicts']
    #     recall_dict = {}
    #     for index in range(batch_size):
    #         pred_boxes = final_pred_dict[index]['pred_boxes']

    #         recall_dict = self.generate_recall_record(
    #             box_preds=pred_boxes,
    #             recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
    #             thresh_list=post_process_cfg.RECALL_THRESH_LIST
    #         )

    #     return final_pred_dict, recall_dict
