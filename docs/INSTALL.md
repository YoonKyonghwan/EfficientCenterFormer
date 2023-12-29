## Installation
This project builds upon [CenterFormer](https://github.com/TuSimple/centerformer). If you encounter any issues during installation or setup, we recommend referring to the detailed instructions provided by CenterFormer. You can also explore solutions to common problems by browsing the issues in their repository.

Our experiments have been successfully tested under the following hardware and software configurations:
- **HardWare**: Orin Nano
- **Software**
    - Jetpack: 5.1.2
        - Python: 3.8.10
        - PyTorch: 2.0.0+nv23.05
        - CUDA: 11.4
        - tensorRT: 8.5.2

### Steps for Installation and Setup

1. **Create a Virtual Environment**
```bash
python -m venv ecf
source ecf/bin/activate
```

2. Run the Setup Script
```bash
. setup_orinNano.sh
```
- The [setup_orinNano.sh](setup_orinNano.sh) script includes commands to temporarily set environment variables necessary for the project.

3. (Optional) Update ~/.bashrc for Permanent Environment Variable Changes:
If you wish to add the project directory to your environment variables permanently, you can append the relevant line to your ~/.bashrc file. This step ensures that the environment variables are set up automatically in every new terminal session.