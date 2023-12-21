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
from det3d.torchie.apis import get_root_logger
from det3d.torchie.apis.train import example_to_device
from det3d.torchie.trainer import load_checkpoint
from det3d.torchie.trainer.utils import all_gather, synchronize
from torch.nn.parallel import DistributedDataParallel
from trt_utils import load_engine, run_trt_engine

def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--config", help="train config file path")
    parser.add_argument("--work_dir", required=True, help="the dir to save logs and models")
    parser.add_argument("--checkpoint", help="the dir to checkpoint which the model read from")
    parser.add_argument("--centerfinder_trt", help="the path of centerfinder trt engine")
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

    batch_size = cfg.data.samples_per_gpu
    data_loader = build_dataloader(
        dataset,
        batch_size=batch_size,
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
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    centerFinder_engine_path = args.centerfinder_trt
    cf_engine = load_engine(centerFinder_engine_path, TRT_LOGGER)
    cf_context = cf_engine.create_execution_context()
    
    
    num_tasks = len(cfg.tasks)
    obj_num = cfg.model['neck']['obj_num']
    x_up = torch.zeros((batch_size, cfg.model['neck']['num_input_features'], 360, 360), dtype=torch.float32).cuda(cfg.local_rank)
    out_scores = torch.zeros((num_tasks, batch_size, obj_num), dtype=torch.float32).cuda(cfg.local_rank)
    out_labels = torch.zeros((num_tasks, batch_size, obj_num), dtype=torch.int32).cuda(cfg.local_rank)
    out_orders = torch.zeros((num_tasks, batch_size, obj_num), dtype=torch.int32).cuda(cfg.local_rank)
    out_masks = torch.zeros((num_tasks, batch_size, obj_num), dtype=torch.bool).cuda(cfg.local_rank)

    for _, data_batch in enumerate(data_loader):
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
                IO_tensors = {
                    "inputs" :
                    {'input_tensor': x},
                    "outputs" :
                    {'x_up': x_up, 'out_scores': out_scores, 'out_labels': out_labels,
                    'out_orders': out_orders, 'out_masks': out_masks}
                }
                run_trt_engine(cf_context, cf_engine, IO_tensors)
                
            with nvtx.annotate("embedding"):
                ct_feat, center_pos_embedding = model.neck.embedding(x_up, out_orders)
            
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
                            "ct_feat": poolformer_output[:, :, idx * obj_num : (idx+1) * obj_num],
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
