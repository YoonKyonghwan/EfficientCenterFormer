cd det3d/ops/iou3d_nms
python setup.py build_ext --inplace

cd ../.. && cd models/ops/
python setup.py build install
python test.py
