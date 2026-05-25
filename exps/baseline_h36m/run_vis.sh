# Baseline 48
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_vis.py --seed 888 --exp-name baseline.txt --dct --layer-norm-axis spatial --with-normalization --num 48 --dim 84 --hidden-dim 84