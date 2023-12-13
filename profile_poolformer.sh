#bin/bash

nsys profile --delay=15 -t nvtx --output=poolformer --force-overwrite=true --stats=true python ./tools/dist_test.py configs/nusc/nuscenes_centerformer_poolformer.py --work_dir work_dirs/nuscenes_poolformer/ --checkpoint work_dirs/nuscenes_poolformer/poolformer.pth
