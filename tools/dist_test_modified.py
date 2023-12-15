import argparse
import copy
import os

import numpy as np
import torch
import yaml
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)

from det3d.torchie.apis.train import example_to_device

from det3d.torchie.trainer import load_checkpoint
from det3d.torchie.trainer.utils import all_gather, synchronize
from torch.nn.parallel import DistributedDataParallel
import pickle 
import time 
import tensorrt as trt

def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", required=True, help="the dir to save logs and models")
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
    parser.add_argument(
        "--txt_result",
        type=bool,
        default=False,
        help="whether to save results to standard KITTI format of txt type",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--speed_test", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--testset", action="store_true")

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args

def run_trt_engine(context, engine, tensors):
    bindings = [None]*engine.num_bindings
    for idx, binding in enumerate(engine):
        tensor_name = engine.get_tensor_name(idx)
        if engine.get_tensor_mode(binding)==trt.TensorIOMode.INPUT:
            bindings[idx] = tensors['inputs'][tensor_name].data_ptr()
            if context.get_tensor_shape(tensor_name):
                context.set_input_shape(tensor_name, tensors['inputs'][tensor_name].shape)
        else:
            bindings[idx] = tensors['outputs'][tensor_name].data_ptr()
    context.execute_v2(bindings=bindings)


def load_engine(engine_filepath, trt_logger):
    with open(engine_filepath, "rb") as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

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

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    
    # print(model)

    if args.testset:
        print("Use Test Set")
        dataset = build_dataset(cfg.data.test)
    else:
        print("Use Val Set")
        dataset = build_dataset(cfg.data.val)

    data_loader = build_dataloader(
        dataset,
        batch_size=cfg.data.samples_per_gpu if not args.speed_test else 1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    # put model on gpus
    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model.cuda(cfg.local_rank),
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            # broadcast_buffers=False,
            find_unused_parameters=True,
        )
    else:
        # model = fuse_bn_recursively(model)
        model = model.cuda()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('parameter size:', pytorch_total_params)

    model.eval()
    mode = "val"

    logger.info(f"work dir: {args.work_dir}")
    if cfg.local_rank == 0:
        prog_bar = torchie.ProgressBar(len(data_loader.dataset) // cfg.gpus)

    detections = {}

    # print(trt.__version__)
    # TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    # centerFinder_engine_path = '/workspace/centerformer/work_dirs/partition/engine/findCenter_folded_op17_v1.trt'
    # cf_engine = load_engine(centerFinder_engine_path, TRT_LOGGER)
    # cf_context = cf_engine.create_execution_context()

    for i, data_batch in enumerate(data_loader):

        device = torch.device(args.local_rank)

        example = example_to_device(data_batch, device, non_blocking=False)
        del data_batch
        with torch.no_grad():
            # outputs = model(example, return_loss=False)
            reader_output = model.reader(example['points'])    
            voxels, coors, shape = reader_output
            x, _ = model.backbone(voxels, coors, len(example['points']), shape)
            
            ct_feat, center_pos_embedding, out_scores, out_labels, out_orders, out_masks = model.neck.find_centers(x)
            # ct_feat2, center_pos_embedding2, out_scores2, out_labels2, out_orders2, out_masks2 = model.neck.find_centers(x)
            # ct_feat = torch.zeros((1, 3000, 256), dtype=torch.float32, device=x.device)
            # center_pos_embedding = torch.zeros((1, 3000, 256), dtype=torch.float32, device=x.device)
            # out_scores = torch.zeros((6, 1, 500), dtype=torch.float32, device=x.device)
            # out_labels = torch.zeros((6, 1, 500), dtype=torch.int32, device=x.device)
            # out_orders = torch.zeros((6, 1, 500), dtype=torch.int32, device=x.device)
            # out_masks = torch.zeros((6, 1, 500), dtype=torch.bool, device=x.device)
            # IO_tensors = {
            #     "inputs" :
            #     {'input_tensor': x},
            #     "outputs" :
            #     {'ct_feat': ct_feat, 'center_pos_embedding': center_pos_embedding,
            #     'out_scores': out_scores, 'out_labels': out_labels,
            #     'out_orders': out_orders, 'out_masks': out_masks}
            # }
            # run_trt_engine(cf_context, cf_engine, IO_tensors)
            
            # print(ct_feat[0][0][:10])
            # print(ct_feat2[0][0][:10])
            # assert False
            
            ct_feat = model.neck.poolformer_forward(ct_feat, center_pos_embedding)
            
            out_dict_list = []
            
            for idx in range(len(cfg.tasks)):
                out_dict = {}
                out_dict.update(
                    {
                        "scores": out_scores[idx],
                        "labels": out_labels[idx],
                        "order": out_orders[idx],
                        "mask": out_masks[idx],
                        "ct_feat": ct_feat[:, :, idx * cfg.test_cfg['obj_num'] : (idx+1) * cfg.test_cfg['obj_num']],
                    }
                )
                out_dict_list.append(out_dict)
                
            preds = model.bbox_head(out_dict_list)
            outputs = model.bbox_head.predict(example, preds, model.test_cfg)
            
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

    if result_dict is not None:
        for k, v in result_dict["results"].items():
            print(f"Evaluation {k}: {v}")

    if args.txt_result:
        assert False, "No longer support kitti"

if __name__ == "__main__":
    main()
