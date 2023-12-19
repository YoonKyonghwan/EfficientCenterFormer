## Installation
Modified from [CenterFormer](https://github.com/TuSimple/centerformer)

Our experiments are tested on the following environments:
- Python: 3.9.18
- PyTorch: 2.0.1+cu118
- CUDA: 11.8
- tensorRT: 8.5

### Installation & Setup
```bash
# basic python libraries
conda create --name effientCF python=3.9
conda activate effientCF
sh setup.sh
```

```bash
# add CenterFormer to PYTHONPATH by adding the following line to ~/.bashrc (change the path accordingly)
export PYTHONPATH="${PYTHONPATH}:PATH_TO_CENTERFORMER"
```

Most of the libaraies are the same as [CenterFormer](https://github.com/TuSimple/centerformer) except for the transformer part. If you run into any issues, you can also refer to their detailed instructions and search from the issues in their repo.