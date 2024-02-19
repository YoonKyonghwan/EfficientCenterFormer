import matplotlib
matplotlib.use("Agg") # remove GTK error
import time
START_TIME = time.time()
CHECK_PREPARATION_TIME = False

import argparse
import copy
import os
import numpy as np
import torch
import tensorrt as trt
import nvtx

from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis.train import example_to_device
from det3d.torchie.trainer import load_checkpoint
from det3d.torchie.trainer.utils import all_gather, synchronize
from trt_utils import load_engine, run_trt_engine
import pickle
import pyscn

def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--eval_mode", help="time or accuracy", default="time", choices=['time', 'accuracy'])
    parser.add_argument("--dataset", help= "nuscenes or waymo", default="nuscenes", choices=['waymo', 'nuscenes'])
    parser.add_argument("--work_dir", help="the dir to save logs and models", default="work_dirs/nuscenes_poolformer_opt")
    parser.add_argument("--model_type", help="baseline or poolformer", default="poolformer", choices=['baseline', 'poolformer'])
    parser.add_argument("--mem_opt", action="store_true", help="whether to use memory optimization for poolformer")
    parser.add_argument("--trt", action="store_true", help="whether to use trt for centerfinder")
    parser.add_argument("--trt_fp16", action="store_true", help="whether to use fp16 for trt")
    parser.add_argument("--centerfinder_trt", help="the path of centerfinder trt engine", default="work_dirs/partition/engine/findCenter_sanitized_fp32.trt")
    parser.add_argument("--backbone_opt", action="store_true", help="whether to use backbone optimization")
    parser.add_argument("--backbone_onnx", help="the path of backbone onnx", default="work_dirs/partition/onnx/poolformer.scn.onnx")
    parser.add_argument("--testset", action="store_true", help="whether to use test set for evaluation")
    parser.add_argument("--verbose", action="store_true", help="Print a thorough diff report, which may result in slower execution")
    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    if args.trt:
        assert args.centerfinder_trt is not None, "Please specify the path of centerfinder trt engine"
    if args.backbone_opt:
        assert args.backbone_onnx is not None, "Please specify the path of backbone onnx"
    
    if args.dataset == "nuscenes":
        if args.model_type == "baseline":
            config = "configs/nusc/nuscenes_centerformer_baseline.py"
            checkpoint = "work_dirs/checkpoint/nuscenes_baseline.pth"
        elif args.model_type == "poolformer":
            config = "configs/nusc/nuscenes_centerformer_poolformer.py"
            checkpoint = "work_dirs/checkpoint/nuscenes_poolformer.pth"
        else:
            print(f"not implemented yet for model type[{args.model_type}]")
            NotImplementedError
    elif args.dataset == "waymo":
        if args.model_type == "baseline":
            config = "configs/waymo/voxelnet/waymo_centerformer_deformable.py"
            checkpoint = "work_dirs/checkpoint/waymo_baseline.pth"
        elif args.model_type == "poolformer":
            config = "configs/waymo/voxelnet/waymo_centerformer_poolformer.py"
            checkpoint = "work_dirs/checkpoint/waymo_poolformer.pth"
        else:
            print(f"not implemented yet for model type[{args.model_type}]")
            NotImplementedError
    else :
        print(f"not implemented yet for other datasets[{args.dataset}]")
        NotImplementedError
    print(f"pth file: {checkpoint}")
        
    cfg = Config.fromfile(config)
    
    if args.eval_mode == "time":
        # nusc
        if args.dataset == "nuscenes":
            cfg.data["val"]["info_path"] = cfg.data_root + "infos_val_time_analysis.pkl"
            cfg.data["val"]["ann_file"] = cfg.data_root + "infos_val_time_analysis.pkl"

        # waymo
        elif args.dataset == "waymo":
            cfg.data["val"]["info_path"] = cfg.data_root + "infos_val_01sweeps_filter_zero_gt_time_analysis.pkl"
            cfg.data["val"]["ann_file"] = cfg.data_root + "infos_val_01sweeps_filter_zero_gt_time_analysis.pkl"
    else: # accuracy
        # nusc
        if args.dataset == "nuscenes":
            cfg.data["val"]["info_path"] = cfg.data_root + "infos_val_accuracy_analysis.pkl"
            cfg.data["val"]["ann_file"] = cfg.data_root + "infos_val_accuracy_analysis.pkl"

        # waymo
        elif args.dataset == "waymo":
            cfg.data["val"]["info_path"] = cfg.data_root + "infos_val_01sweeps_filter_zero_gt.pkl"
            cfg.data["val"]["ann_file"] = cfg.data_root + "infos_val_01sweeps_filter_zero_gt.pkl"
    print(cfg.val_anno)

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    # # init logger before other steps
    # logger = get_root_logger(cfg.log_level)
    # logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    # change center number in model config based on test_cfg
    if 'obj_num' in cfg.test_cfg:
        cfg.model['neck']['obj_num'] = cfg.test_cfg['obj_num']
        print('Use center number {} in inference'.format(cfg.model['neck']['obj_num']))
    if 'score_threshold' in cfg.test_cfg:
        cfg.model['neck']['score_threshold'] = cfg.test_cfg['score_threshold']
        print('Use heatmap score threshold {} in inference'.format(cfg.model['neck']['score_threshold']))

    # set model
    print("set model")
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, checkpoint, map_location="cpu")
    model = model.cuda()
    model.eval()

    print("build dataset")
    # build dataset
    if args.testset:
        print("Use Test Set")
        dataset = build_dataset(cfg.data.test)
    else:
        print("Use Val Set")
        dataset = build_dataset(cfg.data.val)

    batch_size = cfg.data.samples_per_gpu
    data_loader = build_dataloader(
        dataset,
        batch_size=batch_size,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    print(len(data_loader))

    if args.trt:
        # load centerfinder trt engine and allocate buffers
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        centerFinder_engine_path = args.centerfinder_trt
        if args.trt_fp16:
            centerFinder_engine_path = centerFinder_engine_path.replace("fp32", "fp16")
        print(centerFinder_engine_path)
        cf_engine = load_engine(centerFinder_engine_path, TRT_LOGGER)
        cf_context = cf_engine.create_execution_context()
        
        num_tasks = len(cfg.tasks)
        obj_num = cfg.model['neck']['obj_num']
        num_input_features = cfg.model['neck']['num_input_features']

        # ct_feat, center_pos_embedding, out_scores, out_labels, out_orders, out_masks = create_output_tensor(num_tasks, batch_size, obj_num, num_input_features, args.trt_fp16)
        
        # out_labels = torch.zeros((num_tasks, batch_size, obj_num), dtype=torch.int32).cuda()
        # out_orders = torch.zeros((num_tasks, batch_size, obj_num), dtype=torch.int32).cuda()
        # out_masks = torch.zeros((num_tasks, batch_size, obj_num), dtype=torch.bool).cuda()
        # if args.trt_fp16:
        #     ct_feat = torch.zeros((batch_size, num_tasks*obj_num, num_input_features), dtype=torch.float16).cuda()
        #     center_pos_embedding = torch.zeros((batch_size, num_tasks*obj_num, num_input_features), dtype=torch.float16).cuda()
        #     out_scores = torch.zeros((num_tasks, batch_size, obj_num), dtype=torch.float16).cuda()
        # else:
        #     ct_feat = torch.zeros((batch_size, num_tasks*obj_num, num_input_features), dtype=torch.float32).cuda()
        #     center_pos_embedding = torch.zeros((batch_size, num_tasks*obj_num, num_input_features), dtype=torch.float32).cuda()
        #     out_scores = torch.zeros((num_tasks, batch_size, obj_num), dtype=torch.float32).cuda()
    
    if args.backbone_opt:
        inference_type = "fp16"  # fp16 or int8
        backbone_model = pyscn.SCNModel(args.backbone_onnx, inference_type)

    print("start inference")
    prog_bar = torchie.ProgressBar(len(data_loader.dataset))
    detections = {}
    for _, data_batch in enumerate(data_loader):
        example = example_to_device(data_batch, torch.device('cuda'), non_blocking=False)
        example_points = example['points']
        example_metadata = example['metadata']
        del data_batch
        del example
        
        with torch.no_grad():                
            with nvtx.annotate("reader"):
                reader_output = model.reader(example_points)    
                
            with nvtx.annotate("3D_backbone"):
                voxels, coors, shape = reader_output
                if args.backbone_opt:
                    # print("Backbone PySCN")
                    if args.verbose:
                        x_org = model.backbone(voxels, coors, len(example_points), shape)[0]
                    
                    voxels = voxels.detach().cpu().to(dtype=torch.float16).numpy()
                    # voxels = voxels.astype(np.float16)  # (70353, 5)
                    coors = coors.detach().cpu().to(dtype=torch.int32).numpy()
                    # coors = coors.astype(np.int32)      # (70353, 4)
                    shape = shape.tolist()              # [1440, 1440, 40]
                    x_opt = backbone_model.forward(voxels, coors, [41,1440,1440], 0)[0]   # (1, 256, 180, 180)
                    if args.verbose:
                        diff(x_org, x_opt)

                    # Two way for dtype conversion, both demonstrating equivalent accuracy
                    # x = torch.from_numpy(x_opt).to(dtype=torch.float32).cuda()
                    x = torch.from_numpy(x_opt.astype(np.float32)).cuda()
                    # print(f"x_opt:{x_opt.dtype} x:{x.dtype}")
                else:
                    # print("Backbone Torch")
                    x = model.backbone(voxels, coors, len(example_points), shape)[0]        # [1, 256, 180, 180]
            
            if args.model_type == "baseline":
                out_dict_list = model.neck(x)
            elif args.model_type == "poolformer":
                with nvtx.annotate("find_centers"): 
                    if args.trt:
                        if args.verbose:
                            x_org = x.clone().detach()
                        # print(f"input tensor(SCN) type: {x.dtype}")
                        ct_feat, center_pos_embedding, out_scores, out_labels, out_orders, out_masks = create_output_tensor(num_tasks, batch_size, obj_num, num_input_features)
                        IO_tensors = {
                            "inputs" :
                            {
                                'input_tensor': x
                            },
                            "outputs" :
                            {
                                'ct_feat':ct_feat, 'center_pos_embedding': center_pos_embedding,
                                'out_scores': out_scores, 'out_labels': out_labels,
                                'out_orders': out_orders, 'out_masks': out_masks
                            }
                        }
                        run_trt_engine(cf_context, cf_engine, IO_tensors)
                        
                        if args.verbose:
                            ct_feat_org, center_pos_embedding_org, out_scores_org, out_labels_org, out_orders_org, out_masks_org = model.neck.find_centers(x_org)
                            diff(ct_feat_org, ct_feat.clone().detach().cpu().numpy())
                            diff(center_pos_embedding_org, center_pos_embedding.clone().detach().cpu().numpy())
                            diff(out_scores_org, out_scores.clone().detach().cpu().numpy())
                            diff(out_labels_org, out_labels.clone().detach().cpu().numpy())
                            diff(out_orders_org, out_orders.clone().detach().cpu().numpy())
                            diff(out_masks_org, out_masks.clone().detach().cpu().numpy())

                    elif args.mem_opt:
                        ct_feat, center_pos_embedding, out_scores, out_labels, out_orders, out_masks = model.neck.find_centers(x)
                    else:
                        ct_feat, center_pos_embedding, out_dict_list = model.neck.find_centers_baseline(x)
                
                with nvtx.annotate("poolformer_forward"):
                    if args.mem_opt:
                        out_dict_list = model.neck.poolformer(ct_feat, center_pos_embedding, out_scores, out_labels, out_orders, out_masks)
                    else:
                        out_dict_list = model.neck.poolformer_baseline(ct_feat, center_pos_embedding, out_dict_list)
            
            with nvtx.annotate("bbox_head"):
                if args.mem_opt:
                    preds = model.bbox_head(out_dict_list)
                else:
                    preds = model.bbox_head.forward_baseline(out_dict_list)
                
            with nvtx.annotate("post_processing"):
                if args.mem_opt:
                    outputs = model.bbox_head.predict(example_metadata, preds, model.test_cfg)
                else:
                    outputs = model.bbox_head.predict_baseline(example_metadata, preds, model.test_cfg)
            
        for output in outputs:
            token = output["metadata"]["token"]
            for k, v in output.items():
                if k not in [
                    "metadata",
                ]:
                    output[k] = v.cpu()
            detections.update(
                {token: output,}
            )
            
            prog_bar.update()
       
        global CHECK_PREPARATION_TIME 
        if not CHECK_PREPARATION_TIME:
            CHECK_PREPARATION_TIME = True
            print("time for preparation: ", time.time() - START_TIME)

    if args.eval_mode == "time":
        return

    print("synchronize")
    synchronize()

    print("all_gather")
    # torch.cuda.empty_cache()
    all_predictions = all_gather(detections)

    print("save predictions")
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
        
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
        
    with open(os.path.join(args.work_dir, "prediction.pkl"), "wb") as f:
        pickle.dump(predictions, f)


    # print("evaluation")
    # result_dict, _ = dataset.evaluation(copy.deepcopy(predictions), output_dir=args.work_dir, testset=args.testset)

def diff(torch_tensor, cpp_tensor):
    cpp_shape    = ' x '.join(map(str, cpp_tensor.shape))
    torch_shape  = ' x '.join(map(str, torch_tensor.shape))
    print("================ Compare Information =================")
    print(f" CPP     Tensor: {cpp_shape}, {cpp_tensor.dtype}")
    print(f" PyTorch Tensor: {torch_shape}, {torch_tensor.dtype}")

    if np.cumprod(cpp_tensor.shape)[-1] != np.cumprod(torch_tensor.shape)[-1]:
        raise RuntimeError(f"Invalid compare with mismatched shape, {cpp_shape} < ----- > {torch_shape}")

    cpp_tensor   = cpp_tensor.reshape(-1).astype(np.float32)
    torch_tensor = torch_tensor.detach().cpu().numpy().reshape(-1).astype(np.float32)

    diff        = np.abs(cpp_tensor - torch_tensor)
    absdiff_max = diff.max().item()
    print(f"\033[31m[absdiff]: max:{absdiff_max}, sum:{diff.sum().item():.6f}, std:{diff.std().item():.6f}, mean:{diff.mean().item():.6f}\033[0m")
    print(f"CPP:   absmax:{np.abs(cpp_tensor).max().item():.6f}, min:{cpp_tensor.min().item():.6f}, std:{cpp_tensor.std().item():.6f}, mean:{cpp_tensor.mean().item():.6f}")
    print(f"Torch: absmax:{np.abs(torch_tensor).max().item():.6f}, min:{torch_tensor.min().item():.6f}, std:{torch_tensor.std().item():.6f}, mean:{torch_tensor.mean().item():.6f}")
    
    absdiff_p75 = absdiff_max * 0.75
    absdiff_p50 = absdiff_max * 0.50
    absdiff_p25 = absdiff_max * 0.25
    numel       = cpp_tensor.shape[0]
    num_p75     = np.sum(diff > absdiff_p75)
    num_p50     = np.sum(diff > absdiff_p50)
    num_p25     = np.sum(diff > absdiff_p25)
    num_p00     = np.sum(diff > 0)
    num_eq00    = np.sum(diff == 0)
    print(f"[absdiff > m75% --- {absdiff_p75:.6f}]: {num_p75 / numel * 100:.3f} %, {num_p75}")
    print(f"[absdiff > m50% --- {absdiff_p50:.6f}]: {num_p50 / numel * 100:.3f} %, {num_p50}")
    print(f"[absdiff > m25% --- {absdiff_p25:.6f}]: {num_p25 / numel * 100:.3f} %, {num_p25}")
    print(f"[absdiff > 0]: {num_p00 / numel * 100:.3f} %, {num_p00}")
    print(f"[absdiff = 0]: {num_eq00 / numel * 100:.3f} %, {num_eq00}")

    cpp_norm   = np.linalg.norm(cpp_tensor)
    torch_norm = np.linalg.norm(torch_tensor)
    sim        = (np.matmul(cpp_tensor, torch_tensor) / (cpp_norm * torch_norm))
    print(f"[cosine]: {sim * 100:.3f} %")
    print("======================================================")
    return

def create_output_tensor(num_tasks, batch_size, obj_num, num_input_features, trt_fp16=False):
    out_labels = torch.zeros((num_tasks, batch_size, obj_num), dtype=torch.int32).cuda()
    out_orders = torch.zeros((num_tasks, batch_size, obj_num), dtype=torch.int32).cuda()
    out_masks = torch.zeros((num_tasks, batch_size, obj_num), dtype=torch.bool).cuda()
    if trt_fp16:
        ct_feat = torch.zeros((batch_size, num_tasks*obj_num, num_input_features), dtype=torch.float16).cuda()
        center_pos_embedding = torch.zeros((batch_size, num_tasks*obj_num, num_input_features), dtype=torch.float16).cuda()
        out_scores = torch.zeros((num_tasks, batch_size, obj_num), dtype=torch.float16).cuda()
    else:
        ct_feat = torch.zeros((batch_size, num_tasks*obj_num, num_input_features), dtype=torch.float32).cuda()
        center_pos_embedding = torch.zeros((batch_size, num_tasks*obj_num, num_input_features), dtype=torch.float32).cuda()
        out_scores = torch.zeros((num_tasks, batch_size, obj_num), dtype=torch.float32).cuda()
    return ct_feat, center_pos_embedding, out_scores, out_labels, out_orders, out_masks

if __name__ == "__main__":
    main()
