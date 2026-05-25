import argparse
import os, sys
from scipy.spatial.transform import Rotation as R
from thop import profile

import numpy as np
from config  import config
from inner_model import MLP as Model
from patchmlp.PatchMLP import Model as PatchMLP
from datasets.h36m_eval import H36MEval
from utils.misc import rotmat2xyz_torch, rotmat2euler_torch

import torch
from torch.utils.data import DataLoader

from outer_model import HarmonicProcess

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

dct_m,idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)


# mapping adapted from: https://github.com/tancik/fourier-feature-networks
def map_gaussian_fourier_features(x, gmap_size=33, dim=66):
    B = torch.randn((gmap_size, dim))
    x_proj = (2.*np.pi*x.cpu()) @ B.T
    return np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1)

def regress_pred(model, pbar, num_samples, joint_used_xyz, m_p3d_h36, actions=None):
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31]).astype(np.int64)
    joint_equal = np.array([13, 19, 22, 13, 27, 30]).astype(np.int64)

    if actions:
        per_action_errors = {action_name: np.zeros(config.motion.h36m_target_length) for action_name in actions}
        per_action_samples = {action_name: 0 for action_name in actions}

    for (motion_input, motion_target, action_name) in pbar:
        action_name = action_name[0] # retrieve single action name from list of action names (per frame)

        motion_input = motion_input.cuda()
        # fft
        if config.fft:
            motion_target = motion_target.detach().cfloat()
        b,n,c,_ = motion_input.shape
        num_samples += b
        if actions:
            per_action_samples[action_name] += b

        motion_input = motion_input.reshape(b, n, 32, 3)
        motion_input = motion_input[:, :, joint_used_xyz].reshape(b, n, -1)
        outputs = []
        step = config.motion.h36m_target_length_train
        if step == 25:
            num_step = 1
        else:
            num_step = 25 // step + 1
        for idx in range(num_step):
            with torch.no_grad():
                if config.deriv_input:
                    motion_input_ = motion_input.clone()
                    #print(f"0 : {motion_input_.shape}")
                    motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], motion_input_.cuda())
                    #print(f"1 : {motion_input_.shape}")
                else:
                    motion_input_ = motion_input.clone()
                    #print(f"2 : {motion_input_.shape}")

                if config.harm:
                    harmonic_process = HarmonicProcess().cuda()
                    motion_input_ = harmonic_process(motion_input_)

                if config.ffm:
                    motion_input_ = motion_input.clone()
                    motion_input_ = map_gaussian_fourier_features(motion_input_)
                    motion_input_ = torch.tensor(motion_input_).float().cuda()
                if config.fft:
                    motion_input_ = motion_input.clone()
                    #print(f"3 : {motion_input_.shape}")
                    motion_input_ = torch.fft.fft(motion_input_.cfloat())
                    #print(f"4 : {motion_input_.shape}")
                output = model(motion_input_)
                #print(f"5 : {output.shape}")
                if config.ifft:
                    output = torch.fft.ifft(output)[:, :step, :]
                    #print(f"6 : {output.shape}")
                if config.harm:
                    output = harmonic_process.back(output)
                if config.post_dct:
                    output = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], output)[:, :step, :]
                    #print(f"7 : {output.shape}")
                if config.deriv_output:
                    output = output + motion_input[:, -1:, :].repeat(1,step,1)
                    #print(f"8 : {output.shape}")
                else:
                    output = output[:, :step, :]

            output = output.reshape(-1, 22*3)
            output = output.reshape(b,step,-1)
            outputs.append(output)
            motion_input = torch.cat([motion_input[:, step:], output], axis=1)
        motion_pred = torch.cat(outputs, axis=1)[:,:25]

        motion_target = motion_target.detach()
        b,n,c,_ = motion_target.shape

        motion_gt = motion_target.clone()

        motion_pred = motion_pred.detach().cpu()
        pred_rot = motion_pred.clone().reshape(b,n,22,3)
        motion_pred = motion_target.clone().reshape(b,n,32,3)
        motion_pred[:, :, joint_used_xyz] = pred_rot

        tmp = motion_gt.clone()
        tmp[:, :, joint_used_xyz] = motion_pred[:, :, joint_used_xyz]
        motion_pred = tmp
        motion_pred[:, :, joint_to_ignore] = motion_pred[:, :, joint_equal]

        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(motion_pred*1000 - motion_gt*1000, dim=3), dim=2), dim=0)
        if actions:
            per_action_errors[action_name] += mpjpe_p3d_h36.cpu().numpy()
        else:
            m_p3d_h36 += mpjpe_p3d_h36.cpu().numpy()

    if actions:
        for action_name in actions:
            per_action_errors[action_name] /= per_action_samples[action_name]
        return per_action_errors

    m_p3d_h36 = m_p3d_h36 / num_samples
    return m_p3d_h36

def test(config, model, dataloader, vis=False, per_action=False) :

    m_p3d_h36 = np.zeros([config.motion.h36m_target_length])
    titles = np.array(range(config.motion.h36m_target_length)) + 1
    if vis:
        joint_used_xyz = np.arange(0, 28).astype(np.int64)
    else:
        joint_used_xyz = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)

    #if config.fft:
    #    joint_used_xyz = torch.tensor(joint_used_xyz, dtype=torch.complex64)

    num_samples = 0

    pbar = dataloader
    actions = dataloader.dataset._actions
    if per_action:
        m_p3d_h36 = regress_pred(model, pbar, num_samples, joint_used_xyz, m_p3d_h36, actions)
        ret = {action_name: {} for action_name in actions}
        for action_name in actions:
            for j in range(config.motion.h36m_target_length):
                ret[action_name]["#{:d}".format(titles[j])] = [m_p3d_h36[action_name][j], m_p3d_h36[action_name][j]]
        result_summary = {}
        for action_name in actions:
            result_summary[action_name] = [round(ret[action_name][key][0], 1) for key in results_keys]
        return result_summary

    else:
        m_p3d_h36 = regress_pred(model, pbar, num_samples, joint_used_xyz, m_p3d_h36)
        ret = {}
        for j in range(config.motion.h36m_target_length):
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
    #model = PatchMLP(args)

    state_dict = torch.load(args.model_pth)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()

    config.motion.h36m_target_length = config.motion.h36m_target_length_eval
    dataset = H36MEval(config, 'test')

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
        input_ = torch.randn(25, 50, 66).to('cuda')
        flops, params = profile(model, inputs=(input_,), verbose=False)
        print(f"FLOPs: {flops}, Parameter #: {params}")
