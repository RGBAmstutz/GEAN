# GEAN
**GEAN: Gated Efficient Attention Network for 3D Human Motion Prediction**
![image](.github/model_visualization.pdf)
------

## Installation
```
conda env create -f environment.yml
```

## Data Preparation
Identical data preparation to [siMLPe](https://github.com/dulucas/siMLPe)

## Training
```
conda activate gean
cd exps/baseline_h36m/
sh run.sh
```
## Evaluating
### Evaluating newly trained model
```
conda activate gean
cd exps/baseline_h36m/
python test.py --model-pth ./log/snapshot/model-iter-40000.pth
```
### Evaluating paper results (best run)
```
conda activate gean
cd exps/baseline_h36m/
python test.py --model-pth ../../results/h36m-best-40000.pth
```
## Acknowledgements
This code base is built on top of the following repository: 
* https://github.com/dulucas/siMLPe/
