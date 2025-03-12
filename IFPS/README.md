
# Setup

Python 3 dependencies:

* numba 0.58.1
* trimesh 4.1.4
* h5py 3.10.0


```
conda create -n Ifps python=3.8
conda activate Ifps
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install .
```

# Run
```
python example.py
```