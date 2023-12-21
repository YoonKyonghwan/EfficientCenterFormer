python work_dirs/partition/script/export_findCenter_onnx.py \
    --config configs/nusc/nuscenes_centerformer_poolformer.py \
    --checkpoint work_dirs/nuscenes_poolformer/poolformer.pth \
    --onnx_dir work_dirs/partition/onnx \
    --onnx_name findCenter \
    --sanitize

python work_dirs/partition/script/buildEngine.py \
    --onnx_path work_dirs/partition/onnx/findCenter_sanitized.onnx \
    --engine_dir work_dirs/partition/engine