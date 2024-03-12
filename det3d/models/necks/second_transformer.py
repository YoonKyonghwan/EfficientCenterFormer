import torch
import numpy as np

from torch import nn, Tensor
from torch.nn import functional as F

from det3d.torchie.cnn import xavier_init
from det3d.models.utils import Sequential
from det3d.models.utils import Transformer, Deform_Transformer, Poolformer
from det3d.models.utils.tensorRTHelper import run_trt_engine, load_engine

from .. import builder
from ..registry import NECKS
from ..utils import build_norm_layer


class BasicResBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        padding: int = 1,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu2 = nn.ReLU()
        self.downsample = downsample
        if self.downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False),
                nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01))
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample_layer(x)

        out += identity
        out = self.relu2(out)

        return out


@NECKS.register_module
class SECOND_transformer_multitask(nn.Module):
    def __init__(
        self,
        layer_nums,         # [1,2,2]
        ds_num_filters,     # [128,128,256]
        num_input_features, # 128
        transformer_config=None,
        hm_head_layer=2,
        corner_head_layer=2,
        corner=False,
        assign_label_window_size=1,
        # classes=3,
        tasks=[],
        use_gt_training=False,
        norm_cfg=None,
        logger=None,
        init_bias=-2.19,
        score_threshold=0.1,
        obj_num=500,
        parametric_embedding=False,
        **kwargs
    ):
        super(SECOND_transformer_multitask, self).__init__()
        self._down_strides = [1, 2, 2]
        self._up_strides = [1, 2, 4]
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._num_input_features = num_input_features
        self.score_threshold = score_threshold
        self.transformer_config = transformer_config
        self.corner = corner
        self.obj_num = obj_num
        self.use_gt_training = use_gt_training
        self.window_size = assign_label_window_size**2
        self.cross_attention_kernel_size = [3, 3, 3]
        self.batch_id = None
        self.tasks = tasks

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        assert len(self._down_strides) == len(self._layer_nums)
        assert len(self._up_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert self.transformer_config is not None

        in_filters = [
            self._num_input_features,
            self._num_filters[0],
            self._num_filters[1],
        ]

        blocks, deblocks = [], []
        for i, layer_num in enumerate(self._layer_nums):
            block, deblock = self._make_blocks(
                in_filters[i],
                self._num_filters[i],
                layer_num,
                down_stride=self._down_strides[i],
                up_stride=self._up_strides[i],
            )
            blocks.append(block)
            deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

        # heatmap prediction
        self.hm_heads = nn.ModuleList()
        for task in self.tasks:
            hm_head = Sequential()
            for i in range(hm_head_layer - 1):
                hm_head.add(
                    nn.Conv2d(
                        384, # self._num_filters[-1] * 2,
                        64,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                    )
                )
                hm_head.add(build_norm_layer(self._norm_cfg, 64)[1])
                hm_head.add(nn.ReLU())

            hm_head.add(
                nn.Conv2d(64, task['num_class'], kernel_size=3, stride=1, padding=1, bias=True)
            )
            hm_head[-1].bias.data.fill_(init_bias)
            self.hm_heads.append(hm_head)

        # corner heads
        self.corner_heads = nn.ModuleList()
        for task in self.tasks:
            corner_head = Sequential()
            for i in range(corner_head_layer - 1):
                corner_head.add(
                    nn.Conv2d(
                        384, # self._num_filters[-1] * 2,
                        64,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                    )
                )
                corner_head.add(build_norm_layer(self._norm_cfg, 64)[1])
                corner_head.add(nn.ReLU())

            corner_head.add(
                nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True)
            )
            corner_head[-1].bias.data.fill_(init_bias)
            self.corner_heads.append(corner_head)
        
        # Transformer
        self.transformer_layer = Transformer(
            self._num_filters[-1] * 2,
            depth=transformer_config.depth,
            heads=transformer_config.heads,
            dim_head=transformer_config.dim_head,
            mlp_dim=transformer_config.MLP_dim,
            dropout=transformer_config.DP_rate,
            out_attention=transformer_config.out_att,
        )
        self.pos_embedding_type = transformer_config.get(
            "pos_embedding_type", "linear"
        )
        if self.pos_embedding_type == "linear":
            if len(self.tasks)>1:
                self.pos_embedding = nn.Linear(3, self._num_filters[-1] * 2)
            else:
                self.pos_embedding = nn.Linear(2, self._num_filters[-1] * 2)
        elif self.pos_embedding_type == "none":
            self.pos_embedding = None
        else:
            raise NotImplementedError()
        # self.cross_attention_kernel_size = transformer_config.cross_attention_kernel_size
        self.parametric_embedding = parametric_embedding
        if self.parametric_embedding:
            self.query_embed = nn.Embedding(self.obj_num * len(self.tasks), self._num_filters[-1] * 2)
            nn.init.uniform_(self.query_embed.weight, -1.0, 1.0)

        logger.info("Finish second_transformer Initialization")

    def _make_blocks(self, inplanes, planes, num_blocks, down_stride, up_stride):
        # make downsample(residual) block
        stride = down_stride
        layers = [
            BasicResBlock(inplanes, planes, stride=stride, downsample=True) ]
        for _ in range(num_blocks):
            layers.append(
                BasicResBlock(planes, planes))
        block = Sequential(*layers)

        # make upsample(deconv) block
        outplanes = 128
        stride = up_stride
        if stride > 1:
            deblock = Sequential(
                nn.ConvTranspose2d(planes, outplanes, stride, stride=stride, bias=False),
                build_norm_layer(self._norm_cfg, outplanes)[1],
                nn.ReLU(),
            )
        # strid<=1 use normal convolution
        else:
            stride = np.round(1 / stride).astype(np.int64)
            deblock = Sequential(
                nn.Conv2d(planes, outplanes, stride, stride=stride, bias=False),
                build_norm_layer(self._norm_cfg, outplanes)[1],
                nn.ReLU(),
            )
        
        return block, deblock

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, x, example=None):
        x = self.blocks[0](x)       # B * 128 * 360 * 360 [B*C*H*W]
        x_deb = self.deblocks[0](x) # B * 128 * 360 * 360

        x_down1 = self.blocks[1](x) # B * 128 * 180 * 180
        x_down1_deb = self.deblocks[1](x_down1)

        x_down2 = self.blocks[2](x_down1) # B * 128 * 90 * 90
        x_down2_deb = self.deblocks[2](x_down2)
        
        x_up = torch.cat([x_deb, x_down1_deb, x_down2_deb], dim=1)

        order_list = []
        out_dict_list = []
        for idx, task in enumerate(self.tasks):
            # heatmap head
            hm = self.hm_heads[idx](x_up)

            if self.corner and self.corner_heads[0].training:
                corner_hm = self.corner_heads[idx](x_up)
                corner_hm = torch.sigmoid(corner_hm)

            # find top K center location
            hm = torch.sigmoid(hm)
            batch, num_cls, H, W = hm.size()

            scores, labels = torch.max(hm.reshape(batch, num_cls, H * W), dim=1)  # b,H*W

            if self.use_gt_training and self.hm_heads[0].training:
                gt_inds = example["ind"][idx][:, (self.window_size // 2) :: self.window_size]
                gt_masks = example["mask"][idx][
                    :, (self.window_size // 2) :: self.window_size
                ]
                batch_id_gt = torch.from_numpy(np.indices((batch, gt_inds.shape[1]))[0]).to(
                    labels
                )
                scores[batch_id_gt, gt_inds] = scores[batch_id_gt, gt_inds] + gt_masks
                order = scores.sort(1, descending=True)[1]
                order = order[:, : self.obj_num]
                scores[batch_id_gt, gt_inds] = scores[batch_id_gt, gt_inds] - gt_masks
            else:
                order = scores.sort(1, descending=True)[1]
                order = order[:, : self.obj_num]

            scores = torch.gather(scores, 1, order)
            labels = torch.gather(labels, 1, order)
            mask = scores > self.score_threshold
            order_list.append(order)

            out_dict = {}
            out_dict.update(
                {
                    "hm": hm,
                    "scores": scores,
                    "labels": labels,
                    "order": order,
                    "mask": mask,
                    "BEV_feat": x_up,
                    "H": H,
                    "W": W,
                }
            )
            if self.corner and self.corner_heads[0].training:
                out_dict.update({"corner_hm": corner_hm})
            out_dict_list.append(out_dict)

        self.batch_id = torch.from_numpy(np.indices((batch, self.obj_num * len(self.tasks)))[0]).to(
                labels
            )
        order_all = torch.cat(order_list,dim=1)

        ct_feat = (
            x_up.reshape(batch, -1, H * W)
            .transpose(2, 1)
            .contiguous()[self.batch_id, order_all]
        )  # B, 500, C
        
        # create position embedding for each center
        y_coor = order_all // W
        x_coor = order_all - y_coor * W
        y_coor, x_coor = y_coor.to(ct_feat), x_coor.to(ct_feat)
        y_coor, x_coor = y_coor / H, x_coor / W
        pos_features = torch.stack([x_coor, y_coor], dim=2)

        if self.parametric_embedding:
            ct_feat = self.query_embed.weight
            ct_feat = ct_feat.unsqueeze(0).expand(batch, -1, -1)

        # run transformer
        src = torch.cat(
            (
                x_up.reshape(batch, -1, H * W).transpose(2, 1).contiguous(),        # 1, 180**2, 256 / H, W = 180, 180
                x.reshape(batch, -1, (H * W) // 4).transpose(2, 1).contiguous(),    # 1, 90**2, 128 / H, W = 90, 90
                x_down.reshape(batch, -1, (H * W) // 16)                            # 1, 90**2, 256  / H, W = 90, 90
                .transpose(2, 1)
                .contiguous(),
            ),
            dim=1,
        )  # B ,sum(H*W), C
        spatial_shapes = torch.as_tensor(
            [(H, W), (H // 2, W // 2), (H // 4, W // 4)],
            dtype=torch.long,
            device=ct_feat.device,
        )
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )

        if len(self.tasks) > 1:
            task_ids = torch.repeat_interleave(torch.arange(len(self.tasks)).repeat(batch,1), self.obj_num, dim=1).to(pos_features) # B, 500
            pos_features = torch.cat([pos_features, task_ids[:, :, None]],dim=-1)

        transformer_out = self.transformer_layer(
            ct_feat,
            self.pos_embedding,
            src,
            spatial_shapes,
            level_start_index,
            center_pos=pos_features,
        )  # (B,N,C)

        ct_feat = (
            transformer_out["ct_feat"].transpose(2, 1).contiguous()
        )  # B, C, 500

        for idx, task in enumerate(self.tasks):
            out_dict_list[idx]["ct_feat"] = ct_feat[:, :, idx * self.obj_num : (idx+1) * self.obj_num]

        return out_dict_list

    def get_multi_scale_feature(self, center_pos, feats):
        """
        Args:
            center_pos: center coor at the lowest scale feature map [B 500 2]
            feats: multi scale BEV feature 3*[B C H W]
        Returns:
            neighbor_feat: [B 500 K C]
            neighbor_pos: [B 500 K 2]
        """
        kernel_size = self.cross_attention_kernel_size
        batch, num_cls, H, W = feats[0].size()

        center_num = center_pos.shape[1]

        relative_pos_list = []
        neighbor_feat_list = []
        for i, k in enumerate(kernel_size):
            neighbor_coords = torch.arange(-(k // 2), (k // 2) + 1)
            neighbor_coords = torch.flatten(
                torch.stack(torch.meshgrid([neighbor_coords, neighbor_coords]), dim=0),
                1,
            )  # [2, k]
            neighbor_coords = (
                neighbor_coords.permute(1, 0).contiguous().to(center_pos)
            )  # relative coordinate [k, 2]
            neighbor_coords = (
                center_pos[:, :, None, :] // (2**i)
                + neighbor_coords[None, None, :, :]
            )  # coordinates [B, 500, k, 2]
            neighbor_coords = torch.clamp(
                neighbor_coords, min=0, max=H // (2**i) - 1
            )  # prevent out of bound
            feat_id = (
                neighbor_coords[:, :, :, 1] * (W // (2**i))
                + neighbor_coords[:, :, :, 0]
            )  # pixel id [B, 500, k]
            feat_id = feat_id.reshape(batch, -1)  # pixel id [B, 500*k]
            # selected_feat = torch.gather(feats[i].reshape(batch, num_cls,(H*W)//(4**i)).permute(0, 2, 1).contiguous(),1,feat_id)
            selected_feat = (
                feats[i]
                .reshape(batch, num_cls, (H * W) // (4**i))
                .permute(0, 2, 1)
                .contiguous()[self.batch_id.repeat(1, k**2), feat_id]
            )  # B, 500*k, C
            neighbor_feat_list.append(
                selected_feat.reshape(batch, center_num, -1, num_cls)
            )  # B, 500, k, C
            relative_pos_list.append(neighbor_coords * (2**i))  # B, 500, k, 2
            # relative_pos_list.append(F.pad(neighbor_coords*(2**i), (0,1), "constant", i)) # B, 500, k, 3

        neighbor_pos = torch.cat(relative_pos_list, dim=2)  # B, 500, K, 2/3
        neighbor_feats = torch.cat(neighbor_feat_list, dim=2)  # B, 500, K, C
        return neighbor_feats, neighbor_pos

    def get_multi_scale_feature_multiframe(self, center_pos, feats, timeframe):
        """
        Args:
            center_pos: center coor at the lowest scale feature map [B 500 2]
            feats: multi scale BEV feature (3+k)*[B C H W]
            timeframe: timeframe [B,k]
        Returns:
            neighbor_feat: [B 500 K C]
            neighbor_pos: [B 500 K 2]
            neighbor_time: [B 500 K 1]
        """
        kernel_size = self.cross_attention_kernel_size
        batch, num_cls, H, W = feats[0].size()

        center_num = center_pos.shape[1]

        relative_pos_list = []
        neighbor_feat_list = []
        timeframe_list = []
        for i, k in enumerate(kernel_size):
            neighbor_coords = torch.arange(-(k // 2), (k // 2) + 1)
            neighbor_coords = torch.flatten(
                torch.stack(torch.meshgrid([neighbor_coords, neighbor_coords]), dim=0),
                1,
            )  # [2, k]
            neighbor_coords = (
                neighbor_coords.permute(1, 0).contiguous().to(center_pos)
            )  # relative coordinate [k, 2]
            neighbor_coords = (
                center_pos[:, :, None, :] // (2**i)
                + neighbor_coords[None, None, :, :]
            )  # coordinates [B, 500, k, 2]
            neighbor_coords = torch.clamp(
                neighbor_coords, min=0, max=H // (2**i) - 1
            )  # prevent out of bound
            feat_id = (
                neighbor_coords[:, :, :, 1] * (W // (2**i))
                + neighbor_coords[:, :, :, 0]
            )  # pixel id [B, 500, k]
            feat_id = feat_id.reshape(batch, -1)  # pixel id [B, 500*k]
            selected_feat = (
                feats[i]
                .reshape(batch, num_cls, (H * W) // (4**i))
                .permute(0, 2, 1)
                .contiguous()[self.batch_id.repeat(1, k**2), feat_id]
            )  # B, 500*k, C
            neighbor_feat_list.append(
                selected_feat.reshape(batch, center_num, -1, num_cls)
            )  # B, 500, k, C
            relative_pos_list.append(neighbor_coords * (2**i))  # B, 500, k, 2
            timeframe_list.append(
                torch.full_like(neighbor_coords[:, :, :, 0:1], 0)
            )  # B, 500, k
            if i == 0:
                # add previous frame feature
                for frame_num in range(feats[-1].shape[1]):
                    selected_feat = (
                        feats[-1][:, frame_num, :, :, :]
                        .reshape(batch, num_cls, (H * W) // (4**i))
                        .permute(0, 2, 1)
                        .contiguous()[self.batch_id.repeat(1, k**2), feat_id]
                    )  # B, 500*k, C
                    neighbor_feat_list.append(
                        selected_feat.reshape(batch, center_num, -1, num_cls)
                    )
                    relative_pos_list.append(neighbor_coords * (2**i))
                    time = timeframe[:, frame_num + 1].to(selected_feat)  # B
                    timeframe_list.append(
                        time[:, None, None, None]
                        * torch.full_like(neighbor_coords[:, :, :, 0:1], 1)
                    )  # B, 500, k

        neighbor_pos = torch.cat(relative_pos_list, dim=2)  # B, 500, K, 2/3
        neighbor_feats = torch.cat(neighbor_feat_list, dim=2)  # B, 500, K, C
        neighbor_time = torch.cat(timeframe_list, dim=2)  # B, 500, K, 1

        return neighbor_feats, neighbor_pos, neighbor_time
