import numpy as np
from config import config
import torch
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

dct_m,idct_m = get_dct_matrix(config.motion.amass_input_length_dct)
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
        print("configure harmonic process")

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
