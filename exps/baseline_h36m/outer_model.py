import numpy as np
from config import config
import torch
from patchtst.layers.PatchTST_backbone import PatchTST_backbone as patching
from mlp import sh_expand_batch
import torch.nn as nn
import torch.nn.functional as F

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

dct_m,idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

# mapping adapted from: https://github.com/tancik/fourier-feature-networks
def map_gaussian_fourier_features(x, gmap_size=33, dim=66):
    B = torch.randn((gmap_size, dim))
    x_proj = (2.*np.pi*x.cpu()) @ B.T
    return np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1)

def outer_model(h36m_motion_input, model):
    # dct
    if config.deriv_input:
        h36m_motion_input_ = h36m_motion_input.clone()
        h36m_motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], h36m_motion_input_.cuda())
    else:
        h36m_motion_input_ = h36m_motion_input.clone()

    if config.harm:
        pre = h36m_motion_input_
        beta = nn.Parameter(torch.ones(1)).cuda()
        harmonic_process = HarmonicProcess().cuda()
        h36m_motion_input_ = harmonic_process(h36m_motion_input_)

    if config.ffm:
        h36m_motion_input_ = map_gaussian_fourier_features(h36m_motion_input_)
        h36m_motion_input_ = torch.tensor(h36m_motion_input_).float().cuda()

    if config.fft:
        h36m_motion_input_ = torch.fft.fftn(h36m_motion_input_)
        h36m_motion_input_ = h36m_motion_input_.real - h36m_motion_input_.imag

    # inner model
    motion_pred = model(h36m_motion_input_.cuda())

    # idct
    if config.ifft:
        motion_pred = motion_pred.imag + motion_pred.real
        motion_pred = torch.fft.ifftn(motion_pred)
    if config.harm:
        motion_pred = harmonic_process.back(motion_pred)
        motion_pred = beta * motion_pred + (1 - beta) * pre
    if config.post_dct:
        motion_pred = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], motion_pred)
    if config.deriv_output:
        offset = h36m_motion_input[:, -1:].cuda()
        motion_pred = motion_pred[:, :config.motion.h36m_target_length] + offset
    else:
        motion_pred = motion_pred[:, :config.motion.h36m_target_length]

    return motion_pred

class HarmonicProcess(nn.Module):
    def __init__(self, num_joints=22):
        super().__init__()
        self.deg = config.harmonic.deg
        self.num_joints = num_joints
        self.C = (config.harmonic.deg + 1) ** 2  # harmonic channels per joint
        self.harm_proj = nn.Linear(self.num_joints * self.C, self.num_joints * 3)  # 88 → 66

    def reset_parameters(self):
        nn.init.zeros_(self.harm_proj.bias)

    def forward(self, x, dct_m=None, config=None):
        # x: [B, S=66, T]
        B, T, S = x.shape
        J = S // 3  # number of joints

        # reshape into [B*T, J, 3]
        x = x.permute(0, 2, 1).reshape(B * T, J, 3)

        # harmonic expansion
        x = sh_expand_batch(x, deg=self.deg)  # [B*T, J, C]
        x = F.layer_norm(x, x.shape[-1:])  # normalize each joint's harmonic vector

        # back to [B, J*C, T]
        x = x.reshape(B, T, J * self.C)#.transpose(1, 2)

        return x

    def back(self, x):
        x = F.layer_norm(x, x.shape[-1:])
        x = self.harm_proj(x)

        return x
