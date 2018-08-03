# DeepVO: Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks

> **Disclaimer:** This is not an official release. This implementation is based on the ICRA 2017 paper of the same title by Sen Wang, Ronald Clark, Hongkai Wen, and Niki Trigoni. We try to reproduce the results presented in the above paper, while incorporating our own interpretations of the approach, wherever needed.


> *Implementation by:* [*Krishna Murthy*](https://krrish94.github.io) and [*Sarthak Sharma*](https://mila.quebec/en/person/sarthak-sharma/)

> Pretrained models and results will be pushed in due course of time.

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

## Running code

To run the code, from the base directory of the repository, run something like this:

```
python -B main.py -datadir ~/scratch/KITTIOdometry/dataset/ -cachedir ../../scratch/DeepVOCache -nepochs 2 -tensorboardX True -lrScheduler cosine -expID tmp -scf 2 -lr 1e-3 -beta1 0.7 -momentum 0.009 -optMethod sgd -dropout 0.5 -modelType flownet -loadModel cache/flownets_EPE1.951.pth.tar -trainBatch 40 -sbatch True -snapshot 1 -snapshotStrategy none -gradClip 20. -imageWidth 640 -imageHeight 192
```
For a more detailed explanation of parameters, refer to `args.py`.
