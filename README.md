# Inverse Farthest Point Sampling (IFPS) Shell

This repository is the official implementation of our paper, accepted by ACM International Conference on Multimedia Retrieval (**ICMR 2025**):

**Inverse Farthest Point Sampling (IFPS): A Universal and Hierarchical Shell Representation for Discrete Data**

__Authors:__ Nayu Ding, Yujie Lu, Yao Huang, Long Wan, Yan Zhao, Zhijun Fang, Shen Cai*, Lin Gao*.

**Links:** [[Video(Youtube)]](https://youtu.be/uHoOZuhxPY0)

## Method

### Core idea in one sentence
Using only the first $N$ FPS-sampled points, the IFPS shell can be constructed to encapsulate all the original discrete points while employing hierarchical management.

### Graphical Abstract

<p align="center">
 <img src="IFPS/assets/pipeline.jpg" width = "600" alt="ifps" align=center />
</p>

### Implicit Sphere Tree and IFPS Shells with Growing Point Numbers

<p align="center">
 <img src="IFPS/assets/seq_cut.jpg" width = "800" alt="ifps" align=center />
</p>

### Multi-way Tree Structure and IFPS Shells under Various Norms

<p align="center">
 <img src="IFPS/assets/multiway_cut.jpg" width = "800" alt="ifps" align=center />
</p>


### Advantages

1. **Intrinsic Representation**: The centers of bounding volume are FPS-sampled points themselves, with radii computed from these samples.
2. **Hierarchical Representation**: IFPS supports both implicit and explicit *q*-way trees without heuristic partitioning.
3. **Universal Representation**: IPFS extends naturally to discrete data in any dimension, generalizing spheres to hyperspheres (under $\mathbf{L}_2$ norm).

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

## Citation

```bibtex
@inproceedings{Ding2025IFPS,
  title={Inverse Farthest Point Sampling (IFPS): A Universal and Hierarchical Shell Representation for Discrete Data},
  author={Ding, Nayu and Lu, Yujie and Huang, Yao and Wan, Long and Zhao, Yan and Fang, Zhijun and Cai, Shen and Gao, Lin},
  booktitle={ACM International Conference on Multimedia Retrieval (ICMR)}, 
  year={2025},
}
```

