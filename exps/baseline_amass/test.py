import argparse
import os, sys
from scipy.spatial.transform import Rotation as R
from thop import profile

import numpy as np
from tqdm import tqdm
from config  import config
from inner_model import MLP as Model
from utils.misc import rotmat2xyz_torch, rotmat2euler_torch
from datasets.amass_eval import AMASSEval

import torch
from torch.utils.data import DataLoader

results_keys = ['#2', '#4', '#8', '#10', '#14', '#18', '#22', '#25']

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

def regress_pred(pbar, num_samples, m_p3d_h36):

    for (motion_input, motion_target) in pbar:
        motion_input = motion_input.cuda()
        b,n,c = motion_input.shape
        num_samples += b

        motion_input = motion_input.reshape(b, n, 18, 3)
        motion_input = motion_input.reshape(b, n, -1)
        outputs = []
        step = config.motion.amass_target_length_train
        if step == 25:
            num_step = 1
        else:
            num_step = 25 // step + 1
        for idx in range(num_step):
            with torch.no_grad():
                if config.deriv_input:
                    if config.deriv_input:
                        motion_input_ = motion_input.clone()
                        motion_input_ = torch.matmul(dct_m, motion_input_.cuda())
                        motion_input_ = motion_input_[:, -config.motion.amass_input_length:]
                    else:
                        motion_input_ = motion_input.clone()

                    if config.ffm:
                        motion_input_ = motion_input.clone()
                        motion_input_ = map_gaussian_fourier_features(motion_input_)
                        motion_input_ = torch.tensor(motion_input_).float().cuda()
                    if config.fft:
                        motion_input_ = motion_input.clone()
                        motion_input_ = torch.fft.fft(motion_input_.cfloat())
                    output = model(motion_input_)
                    if config.ifft:
                        output = torch.fft.ifft(output)[:, :step, :]
                    if config.post_dct:
                        output = torch.matmul(idct_m[:, :config.motion.amass_input_length, :], output)[:, :step, :]
                    if config.deriv_output:
                        output = output + motion_input[:, -1:, :].repeat(1,step,1)
                    else:
                        output = output[:, :step, :]

                output = model(motion_input_)
                output = torch.matmul(idct_m, output)[:, :step, :]
                if config.deriv_output:
                    output = output + motion_input[:, -1:, :].repeat(1,step,1)

            output = output.reshape(-1, 18*3)
            output = output.reshape(b,step,-1)
            outputs.append(output)
            motion_input = torch.cat([motion_input[:, step:], output], axis=1)
        motion_pred = torch.cat(outputs, axis=1)[:,:25]

        b,n,c = motion_target.shape
        motion_target = motion_target.detach().reshape(b, n, 18, 3)
        motion_gt = motion_target.clone()

        motion_pred = motion_pred.detach().cpu()
        motion_pred = motion_pred.reshape(b, n, 18, 3)

        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(motion_pred*1000 - motion_gt*1000, dim=3), dim=2), dim=0)
        m_p3d_h36 += mpjpe_p3d_h36.cpu().numpy()
    m_p3d_h36 = m_p3d_h36 / num_samples
    return m_p3d_h36

def test(config, model, dataloader, vis=False, per_action=False) :

    m_p3d_h36 = np.zeros([config.motion.amass_target_length])
    titles = np.array(range(config.motion.amass_target_length)) + 1

    num_samples = 0

    pbar = dataloader
    m_p3d_h36 = regress_pred(pbar, num_samples, m_p3d_h36)
    ret = {}
    for j in range(config.motion.amass_target_length):
        ret["#{:d}".format(titles[j])] = [m_p3d_h36[j], m_p3d_h36[j]]
    return [round(ret[key][0], 1) for key in results_keys]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model-pth', type=str, default=None, help='=encoder path')
    parser.add_argument('--per-action', action='store_true', help='=per action MPJPE')

    # PatchMLP parser arguments
    parser.add_argument('--seq_len', type=int, help='=patchMLP sequence length')
    parser.add_argument('--pred_len', type=int, help='=patchMLP predict length')
    parser.add_argument('--use_norm', action='store_true', help='=patchMLP normalization')
    parser.add_argument('--d_model', type=int, help='=patchMLP model dimension')
    parser.add_argument('--e_layers', type=int, help='=patchMLP number of encoder layers')
    parser.add_argument('--enc_in', type=int, help='=patchMLP encoder input size')

    # efficient attention
    parser.add_argument('--att_out', action='store_true', help='=outer attention')

    args = parser.parse_args()
    per_action = args.per_action
    config.att_out = args.att_out

    model = Model(config)

    state_dict = torch.load(args.model_pth)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()

    config.motion.amass_target_length = config.motion.amass_target_length_eval
    dataset = AMASSEval(config, 'test')

    shuffle = False
    sampler = None
    train_sampler = None
    dataloader = DataLoader(dataset, batch_size=128,
                            num_workers=1, drop_last=False,
                            sampler=sampler, shuffle=shuffle, pin_memory=True)

    if per_action:
        results = test(config, model, dataloader, per_action=per_action)
        for action_name, result in results.items():
            print(f"Action: {action_name}, MPJPE: {result}")
    else:
        print(test(config, model, dataloader))
        input_ = torch.randn(25, config.motion.amass_input_length, config.motion.dim).to('cuda')
        flops, params = profile(model, inputs=(input_,), verbose=False)
        print(f"FLOPs: {flops}, Parameter #: {params}")
