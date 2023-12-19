import tensorrt as trt

import argparse
from tools.trt_utils import build_engine_onnx, save_engine
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", help="the path for onnx (including file_name)")
    parser.add_argument("--engine_dir", help="the dir to save engine file (save as onnx_name.trt)")
    parser.add_argument("--fp16", action='store_true', help="whether to use fp16")
    parser.add_argument("--verbose", action='store_true', help="whether to use verbose mode")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # build engine from onnx
    onnx_path = args.onnx_path
    serialized_engine = build_engine_onnx(onnx_path, fp16=args.fp16, verbose=args.verbose)
    
    # save engine
    onnx_file_name = onnx_path.split('/')[-1].split('.')[0]
    if args.fp16:
        engine_file_name = onnx_file_name + '_fp16.trt'
    else:
        engine_file_name = onnx_file_name + '_fp32.trt'
    engine_path = os.path.join(args.engine_dir, engine_file_name)
    save_engine(serialized_engine, engine_path)
    
        
if __name__ == '__main__':
    main()