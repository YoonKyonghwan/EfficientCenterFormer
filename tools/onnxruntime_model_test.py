import argparse
import copy
import os
import numpy as np
import torch
import nvtx
import onnxruntime as ort
import tensorrt as trt

from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import get_root_logger
from det3d.torchie.apis.train import example_to_device
from det3d.torchie.trainer import load_checkpoint
from det3d.torchie.trainer.utils import all_gather, synchronize
from torch.nn.parallel import DistributedDataParallel
from tools.trt_utils import load_engine, run_trt_engine

def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--config", help="train config file path", default="configs/nusc/nuscenes_centerformer_poolformer.py")
    parser.add_argument("--work_dir", required=True, help="the dir to save logs and models", default="work_dirs/nuscenes_poolformer")
    parser.add_argument("--checkpoint", help="the dir to checkpoint which the model read from", default="work_dirs/nuscenes_poolformer/poolformer.pth")
    parser.add_argument("--centerfinder_trt", help="the path of centerfinder trt", default="work_dirs/partition/engine/findCenter_sanitized_fp32.trt")
    parser.add_argument("--gpus", type=int, default=1, help="number of gpus to use " "(only applicable to non-distributed training)")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--testset", action="store_true")

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        cfg.gpus = torch.distributed.get_world_size()
    else:
        cfg.gpus = args.gpus

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info("Distributed testing: {}".format(distributed))
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    # change center number in model config based on test_cfg
    if 'obj_num' in cfg.test_cfg:
        cfg.model['neck']['obj_num'] = cfg.test_cfg['obj_num']
        print('Use center number {} in inference'.format(cfg.model['neck']['obj_num']))
    if 'score_threshold' in cfg.test_cfg:
        cfg.model['neck']['score_threshold'] = cfg.test_cfg['score_threshold']
        print('Use heatmap score threshold {} in inference'.format(cfg.model['neck']['score_threshold']))

    # set model
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint, map_location="cpu")

    # build dataset
    if args.testset:
        print("Use Test Set")
        dataset = build_dataset(cfg.data.test)
    else:
        print("Use Val Set")
        dataset = build_dataset(cfg.data.val)

    data_loader = build_dataloader(
        dataset,
        batch_size=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    # put model on gpus
    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model.cuda(cfg.local_rank),
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            find_unused_parameters=True,
        )
    else:
        model = model.cuda()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('parameter size:', pytorch_total_params)

    model.eval()

    if cfg.local_rank == 0:
        prog_bar = torchie.ProgressBar(len(data_loader.dataset) // cfg.gpus)

    detections = {}

    # load centerfinder trt engine and allocate buffers
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_sess = ort.InferenceSession(args.centerfinder_onnx, providers=providers)
    output_names = [output.name for output in ort_sess.get_outputs()]

    for i, data_batch in enumerate(data_loader):
        device = torch.device(args.local_rank)

        example = example_to_device(data_batch, device, non_blocking=False)
        example_points = example['points']
        example_metadata = example['metadata']
        del data_batch
        del example
        
        with torch.no_grad():
            with nvtx.annotate("reader"):
                reader_output = model.reader(example_points)    
                
            with nvtx.annotate("3D_backbone"):
                voxels, coors, shape = reader_output
                x, _ = model.backbone(voxels, coors, len(example_points), shape)
            
            with nvtx.annotate("find_centers"):
                # ct_feat, center_pos_embedding, out_scores, out_labels, out_orders, out_masks = model.neck.find_centers(x)
                # IO_tensors = {
                #     "inputs" :
                #     {'input_tensor': x},
                #     "outputs" :
                #     {'ct_feat': ct_feat, 'center_pos_embedding': center_pos_embedding,
                #     'out_scores': out_scores, 'out_labels': out_labels,
                #     'out_orders': out_orders, 'out_masks': out_masks}
                # }
                # run_trt_engine(cf_context, cf_engine, IO_tensors)
                ct_feat, center_pos_embedding, out_scores, out_labels, out_orders, out_masks = ort_sess.run(output_names, {'input_tensor': x.cpu().numpy()})
            
                ct_feat = torch.from_numpy(ct_feat).cuda()
                center_pos_embedding = torch.from_numpy(center_pos_embedding).cuda()
                out_scores = torch.from_numpy(out_scores).cuda()
                out_labels = torch.from_numpy(out_labels).cuda()
                out_orders = torch.from_numpy(out_orders).cuda()
                out_masks = torch.from_numpy(out_masks).cuda()
                
            with nvtx.annotate("poolformer_forward"):
                poolformer_output = model.neck.poolformer_forward(ct_feat, center_pos_embedding)
            
                out_dict_list = []
                
                for idx in range(len(cfg.tasks)):
                    out_dict = {}
                    out_dict.update(
                        {
                            "scores": out_scores[idx],
                            "labels": out_labels[idx],
                            "order": out_orders[idx],
                            "mask": out_masks[idx],
                            "ct_feat": poolformer_output[:, :, idx * cfg.test_cfg['obj_num'] : (idx+1) * cfg.test_cfg['obj_num']],
                        }
                    )
                    out_dict_list.append(out_dict)
            
            with nvtx.annotate("bbox_head"):
                preds = model.bbox_head(out_dict_list)
                
            with nvtx.annotate("post_processing"):
                outputs = model.bbox_head.predict(example_metadata, preds, model.test_cfg)
            
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
            if args.local_rank == 0:
                prog_bar.update()

    synchronize()

    # torch.cuda.empty_cache()
    all_predictions = all_gather(detections)

    if args.local_rank != 0:
        return

    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    result_dict, _ = dataset.evaluation(copy.deepcopy(predictions), output_dir=args.work_dir, testset=args.testset)

    # if result_dict is not None:
    #     for k, v in result_dict["results"].items():
    #         print(f"Evaluation {k}: {v}")

if __name__ == "__main__":
    main()
