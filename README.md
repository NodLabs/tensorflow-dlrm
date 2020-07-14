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
