## Installation
Modified from [CenterFormer](https://github.com/TuSimple/centerformer)

Our experiments are tested on the following environments:
- H/W: Orin Nano
- Jetpack: 5.1.2
    - Python: 3.8.10
    - PyTorch: 2.0.0+nv23.05
    - CUDA: 11.4
    - tensorRT: 8.5.2

### Installation & Setup
```bash
# create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
sh setup.sh
```

```bash
# add the project directory to PYTHONPATH (by adding the following line to ~/.bashrc)
# change the path accordingly
export PYTHONPATH="${PYTHONPATH}:PATH_TO_PRJECT"
```

Most of the libaraies are the same as [CenterFormer](https://github.com/TuSimple/centerformer). If you run into any issues, you can also refer to their detailed instructions and search from the issues in their repo.