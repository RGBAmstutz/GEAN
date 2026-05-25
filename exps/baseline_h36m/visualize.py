import argparse
import os, sys
from scipy.spatial.transform import Rotation as R
from thop import profile

import numpy as np
from config  import config
from inner_model import MLP as SphericalMultiKAN
from datasets.h36m_eval import H36MEval
from utils.misc import rotmat2xyz_torch, rotmat2euler_torch
import utils.weight_visualization as viz

import torch
import netron

results_keys = ['#2', '#4', '#8', '#10', '#14', '#18', '#22', '#25']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model-pth', type=str, default=None, help='=encoder path')

    # PatchMLP parser arguments
    parser.add_argument('--seq_len', type=int, help='=patchMLP sequence length')
    parser.add_argument('--pred_len', type=int, help='=patchMLP predict length')
    parser.add_argument('--use_norm', action='store_true', help='=patchMLP normalization')
    parser.add_argument('--d_model', type=int, help='=patchMLP model dimension')
    parser.add_argument('--e_layers', type=int, help='=patchMLP number of encoder layers')
    parser.add_argument('--enc_in', type=int, help='=patchMLP encoder input size')

    args = parser.parse_args()

    model = SphericalMultiKAN(config)

    state_dict = torch.load(args.model_pth, weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()

print(model)
