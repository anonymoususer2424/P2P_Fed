# Experimental Setup & Run Experiments

## Experimental Setup
Our experiments rely on Docker. Before conducting the experiments, the following environment setup is assumed. \
In the description, a ```node``` refers to a single container, while a ```machine``` refers to a physical device that can have one or multiple nodes.


Our SW and HW setup of each ```node``` is as follows.
- SW: Ubuntu 20.04.6, Python 3.10.9, PyTorch 2.0.0, CUDA 11.7, Docker 24.0.5, NVIDIA Driver 525.147.05
- HW (per node): CPU: 1 core, Memory: 6 GB, GPU (VRAM): 1.6 GB
### Environment Variable Setup
You need to set the various environment variables in config.json according to your environment.
If you are using multiple machines, all settings except for ```DIR``` and ```ETC``` must be the same across all machines.

### Build the ```Dockerfile```
```shell
# In the project directory
docker build --force-rm -t im4P2PFed . # The image name must match the one in config.json.
```

### Activate ```MPS``` (NVIDIA's Multi-Process Service)
```MPS```[[1]](#1) is a runtime service that allows multiple processes to run simultaneously on a single GPU. ```MPS``` facilitates the allocation of GPU resources to each process and supports independent memory spaces.
To activate this, we refer to [this](https://github.com/emptyinteger/Nvidia-MPS-Docker-Pytorch-ShellScript).


### Activate ```IFB``` (Intermediate Functional Block)
Considering that the communication bandwidth between two nodes on the same machine is much higher than the actual network conditions, we have applied a traffic bandwidth limit to the containers. For this, ```IFB``` must be running on the host OS of the machine.
```shell
# In host OS
sudo modprobe ifb
```
The settings within the container are in tc-script.sh, and they are automatically applied when the container is created via the ```Dockerfile```.

### Data Configuration
Refer to the [```data/README```](../data/README.md) to create data partitions. If you are using multiple machines, the data partitions must be identical, so we recommend creating them on one machine and then using ```scp``` to distribute them.

### Conda Environment
Our experiments use accuracy per wall clock time as the results, so we cannot conduct evaluations during runtime. Instead, we save paramters of model and perform evaluations after the entire workflow is completed. Therefore, we need an environment for evaluation. \
While we provide this environment through ```requirements.txt```, following these specific library versions is not mandatory as they do not impact performance. 

```shell
# In the project directory
pip install -r requirements.txt
```

## References
<a id="1">[1]</a> 
https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf.