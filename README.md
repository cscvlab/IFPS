# Inverse Farthest Point Sampling (IFPS) Shell

This repository is the official implementation of our **ICMR 2025** paper:

**Inverse Farthest Point Sampling (IFPS): A Universal and Hierarchical Shell Representation for Discrete Data**

__Authors:__ Nayu Ding, Yujie Lu, Yao Huang, Long Wan, Yan Zhao, Zhijun Fang, Shen Cai*, Lin Gao*.

**Links:** [[Video(Youtube)]](https://youtu.be/uHoOZuhxPY0)

## Method

**Core idea in one sentence**. Using only the first $N$ FPS-sampled points, the IFPS shell can be constructed to encapsulate all the original discrete points while employing hierarchical management.

### Pipeline

<p align="center">
 <img src="IFPS/assets/pipeline.jpg" width = "800" alt="ifps" align=center />
</p>

![ifps](IFPS/assets/pipeline.jpg)


### Implicit Sphere Tree and IFPS Shells with Growing Point Numbers

![ifps](IFPS/assets/seq_cut.jpg)


### Multi-way Tree Structure and IFPS Shells under Various Norms

![ifps](IFPS/assets/multiway_cut.jpg)


## Setup

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

## DataSet
You can use the example dataset in IFPS/dataset/thingi32_normalization/ or you can put your custom datasets in IFPS/dataset/ directory.

## Configurations
All configurable settings are accessible within the IFPS/utils/options.py

## Run
```
python example.py
```

