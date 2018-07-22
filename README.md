# DeepVO: Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks

> **Disclaimer:** This is not an official release. This implementation is based on the ICRA 2017 paper of the same title by Sen Wang, Ronald Clark, Hongkai Wen, and Niki Trigoni. We try to reproduce the results presented in the above paper, while incorporating our own interpretations of the approach, wherever needed.


> *Implementation by:* *Sarthak Sharma* and [*Krishna Murthy*](https://krrish94.github.io)

## Installation Instructions

This is a `PyTorch` implementation. We assume `PyTorch` and dependencies are setup. This code has been tested on `PyTorch 0.4` with `CUDA 9.0` and `CUDNN <VERSION>`.

Dependencies: `scipy`, `scikit-image`, `matplotlib`, `tqdm`, and `natsort`.
We also use `tensorboardX` for visualization purposes.

`scipy`, `scikit-image`, `matplotlib`, and `natsort` can be installed using standard pip or conda commands.

#### tqdm (pip)
```
pip install --user tqdm
```

#### tqdm (conda)
```
conda install -c conda-forge tqdm
```

#### tensorboardX (pip)
```
pip install tensorboardX
```

#### tensorboardX (build from source)
```
pip install git+https://github.com/lanpa/tensorboard-pytorch
```
