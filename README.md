# tensorflow-dlrm
This is Nod's Tensorflow version of DLRM which is based on [**OpenRec**](http://www.openrec.ai/) DLRM model. We extract the Openrec DRML source code and fixed some bugs in their model definition to make it work with tensorflow-gpu==2.2 and python3.7


## Install tensorflow-dlrm from source code ##

First, clone noddlrm using `git`:

```sh
git clone https://github.com/NodLabs/tensorflow-dlrm
```

Then, `cd` to the tensorflow-dlrm folder and run the install command(if you want to install 
noddlrm to your python lib):

```sh
cd tensorflow-dlrm
python setup.py install
```
Now you have installed noddlrm to you system.

## Dataset download

All datasets can be downloaded from Google drive [here](https://drive.google.com/drive/folders/1taJ91txiMAWBMUtezc_N5gaYuTEpvW_e?usp=sharing).
In our example, we use the dataset criteo.

## Training and get the saved model
Edit the  dlrm_criteo_gpu/tpu.py to use your dataset criteo path
Then run the example script we have provided.
```sh
cd tensorflow-dlrm/
export PYTHONPATH="$PWD"
python3  dlrm_criteo_gpu.py
# python3  dlrm_criteo_tpu.py
```
## Outputs ##
### GPU ###
```sh
python3  dlrm_criteo_gpu.py

2020-07-14 08:10:32.701182: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2020-07-14 08:10:32.729195: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-14 08:10:32.729514: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: GeForce RTX 2080 computeCapability: 7.5
coreClock: 1.59GHz coreCount: 46 deviceMemorySize: 7.79GiB deviceMemoryBandwidth: 417.29GiB/s
2020-07-14 08:10:32.729734: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/chi/bin/vulkansdk/x86_64/lib::/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
2020-07-14 08:10:32.730158: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcublas.so.10'; dlerror: libcublas.so.10: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/chi/bin/vulkansdk/x86_64/lib::/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
2020-07-14 08:10:32.730298: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcufft.so.10'; dlerror: libcufft.so.10: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/chi/bin/vulkansdk/x86_64/lib::/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
2020-07-14 08:10:32.730501: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcurand.so.10'; dlerror: libcurand.so.10: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/chi/bin/vulkansdk/x86_64/lib::/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
2020-07-14 08:10:32.730632: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcusolver.so.10'; dlerror: libcusolver.so.10: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/chi/bin/vulkansdk/x86_64/lib::/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
2020-07-14 08:10:32.730848: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcusparse.so.10'; dlerror: libcusparse.so.10: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/chi/bin/vulkansdk/x86_64/lib::/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
2020-07-14 08:10:32.766959: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-07-14 08:10:32.766987: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1598] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2020-07-14 08:10:32.768812: I tensorflow/core/platform/cpu_feature_guard.cc:143] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-14 08:10:32.799122: I tensorflow/core/platform/profile_utils/cpu_utils.cc:102] CPU Frequency: 2899885000 Hz
2020-07-14 08:10:32.799874: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7f2204000b20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-14 08:10:32.799887: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-07-14 08:10:32.801441: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-07-14 08:10:32.801453: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1108]      
WARNING:tensorflow:From /home/chi/nnc_env/lib/python3.7/site-packages/tensorflow/python/ops/linalg/linear_operator_lower_triangular.py:158: calling LinearOperator.__init__ (from tensorflow.python.ops.linalg.linear_operator) with graph_parents is deprecated and will be removed in a future version.
Instructions for updating:
Do not pass `graph_parents`.  They will  no longer be used.
Iter: 0, Loss: 0.24, AUC: 0.5614          
Iter: 100, Loss: 0.19, AUC: 0.6755           
Iter: 200, Loss: 0.17, AUC: 0.6976           
Iter: 300, Loss: 0.17, AUC: 0.7037           
Iter: 400, Loss: 0.17, AUC: 0.7062           
Iter: 500, Loss: 0.17, AUC: 0.7079           
Iter: 600, Loss: 0.17, AUC: 0.7080           
Iter: 700, Loss: 0.17, AUC: 0.7095           
Iter: 800, Loss: 0.17, AUC: 0.7099           
Iter: 900, Loss: 0.17, AUC: 0.7103           
2020-07-14 08:21:00.611816: W tensorflow/python/util/util.cc:329] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
WARNING:tensorflow:From /home/chi/nnc_env/lib/python3.7/site-packages/tensorflow/python/ops/resource_variable_ops.py:1817: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.

```
### TPU ###
```sh
python3  dlrm_criteo_tpu.py

2020-07-14 15:16:10.152558: I tensorflow/core/platform/cpu_feature_guard.cc:143] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-14 15:16:10.179204: I tensorflow/core/platform/profile_utils/cpu_utils.cc:102] CPU Frequency: 2300000000 Hz
2020-07-14 15:16:10.183868: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x38c1f20 initialized for platform Host (this does not guarantee that XLA will be used).
 Devices:
2020-07-14 15:16:10.183924: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-07-14 15:16:10.232148: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:301] Initialize GrpcChannelCache for job worker -> {0 -> 10.240.1.2:8470}
2020-07-14 15:16:10.232200: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:301] Initialize GrpcChannelCache for job localhost -> {0 -> localhost:31017}
2020-07-14 15:16:10.255055: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:301] Initialize GrpcChannelCache for job worker -> {0 -> 10.240.1.2:8470}
2020-07-14 15:16:10.255110: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:301] Initialize GrpcChannelCache for job localhost -> {0 -> localhost:31017}
2020-07-14 15:16:10.259154: I tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:390] Started server with target: grpc://localhost:31017
All devices:  [LogicalDevice(name='/job:worker/replica:0/task:0/device:TPU:7', device_type='TPU'), LogicalDevice(name='/job:worker/replica:0/task:0/device:TPU:6', device_type='TP
U'), LogicalDevice(name='/job:worker/replica:0/task:0/device:TPU:5', device_type='TPU'), LogicalDevice(name='/job:worker/replica:0/task:0/device:TPU:4', device_type='TPU'), Logic
alDevice(name='/job:worker/replica:0/task:0/device:TPU:0', device_type='TPU'), LogicalDevice(name='/job:worker/replica:0/task:0/device:TPU:1', device_type='TPU'), LogicalDevice(n
ame='/job:worker/replica:0/task:0/device:TPU:2', device_type='TPU'), LogicalDevice(name='/job:worker/replica:0/task:0/device:TPU:3', device_type='TPU')]
WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/tensorflow/python/ops/linalg/linear_operator_lower_triangular.py:158: calling LinearOperator.__init__ (from tensorf
low.python.ops.linalg.linear_operator) with graph_parents is deprecated and will be removed in a future version.
Instructions for updating:
Do not pass `graph_parents`.  They will  no longer be used.
WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/tensorflow/python/ops/linalg/linear_operator_lower_triangular.py:158: calling LinearOperator.__init__ (from tensorf
low.python.ops.linalg.linear_operator) with graph_parents is deprecated and will be removed in a future version.
Instructions for updating:
Do not pass `graph_parents`.  They will  no longer be used.
Iter: 0, Loss: 0.22, AUC: 0.4884          
Iter: 100, Loss: 0.19, AUC: 0.6258           
Iter: 200, Loss: 0.18, AUC: 0.6673           
Iter: 300, Loss: 0.18, AUC: 0.6827           
Iter: 400, Loss: 0.17, AUC: 0.6939           
Iter: 500, Loss: 0.17, AUC: 0.7030           
Iter: 600, Loss: 0.17, AUC: 0.7066           
Iter: 700, Loss: 0.17, AUC: 0.7079           
Iter: 800, Loss: 0.17, AUC: 0.7100           
Iter: 900, Loss: 0.17, AUC: 0.7107           
2020-07-14 15:22:10.259060: W tensorflow/python/util/util.cc:329] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/tensorflow/python/ops/resource_variable_ops.py:1817: calling BaseResourceVariable.__init__ (from tensorflow.python.
ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/tensorflow/python/ops/resource_variable_ops.py:1817: calling BaseResourceVariable.__init__ (from tensorflow.python.
ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
```









