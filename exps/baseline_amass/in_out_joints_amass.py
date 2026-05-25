import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from config import config
from inner_model import MLP as Model
from datasets.amass_eval import AMASSEval

def get_dct_matrix(N):
    """ Generate DCT and IDCT matrices """
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 0.5) * k / N)
    idct_m = np.linalg.inv(dct_m)
    assert np.allclose(np.dot(dct_m, idct_m), np.eye(N), atol=1e-6), "assert DCT and IDCT matrices are proper inverses."
    return dct_m, idct_m


def regress_pred(model, dataloader, min_frame_num, max_frame_num, dct_m, idct_m, actions=None):
    joint_used_xyz = np.arange(0, 28).astype(np.int64)
    motion_in_poses = []
    motion_out_poses = []
    motion_model_inputs = []

    # collect motion inputs
    for i, (dataloader_input, _, _) in enumerate(dataloader):
        if i < min_frame_num:
            continue
        if i >= max_frame_num:
            break
        dataloader_input = dataloader_input.cuda()
        b, n, c, _ = dataloader_input.shape
        dataloader_input = dataloader_input.reshape(b, n, 32, 3)
        dataloader_input = dataloader_input[:, :, joint_used_xyz].reshape(b, n, -1)

        motion_model_inputs.append(dataloader_input)

    # concatenate all motion inputs along the batch dimension
    motion_model_inputs = torch.cat(motion_model_inputs, dim=0)

    # ensure motion inputs are on the same device and correct type
    motion_model_inputs = motion_model_inputs.to(torch.float32).cuda()

    # Apply DCT
    motion_input_dct = torch.matmul(torch.tensor(dct_m).float().cuda(), motion_model_inputs)

    # perform prediction using model
    with torch.no_grad():
        motion_model_output = model(motion_input_dct)

        # Apply IDCT
        motion_model_output = torch.matmul(torch.tensor(idct_m).float().cuda(), motion_model_output)
        motion_model_output = motion_model_output.reshape(-1, n, 28, 3)  # Ensure output is properly shaped

    # convert tensors to numpy arrays and format output
    input_poses = motion_model_inputs.detach().cpu().numpy()
    motion_model_output = motion_model_output.detach().cpu().numpy()

    # reshape input poses to match delta_output if necessary
    input_poses = input_poses.reshape(-1, n, 28, 3)
    # add model outputs (vector displacements) to inputs
    output_poses = input_poses + motion_model_output
    # convert final_output_poses back to ndarray
    output_poses = np.array(output_poses)
    # collect motion in poses and motion out poses
    for batch in range(len(output_poses)):
        if len(motion_in_poses) < max_frame_num - min_frame_num:
            reshaped_input = input_poses[batch].reshape(n, 28, 3)
            motion_in_poses.append(reshaped_input[0].tolist())
        if len(motion_out_poses) < max_frame_num - min_frame_num:
            reshaped_output = output_poses[batch].reshape(n, 28, 3)
            motion_out_poses.append(reshaped_output[0].tolist())

    return motion_in_poses, motion_out_poses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-pth', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dim', type=int, default=84, help='=number of dimensions')
    parser.add_argument('--hidden-dim', type=int, default=84, help='=number of hidden dimensions')
    parser.add_argument('--min-frame', type=int, default=0, help='minimum frame count to be processed')
    parser.add_argument('--max-frame', type=int, default=20, help='maximum frame count to be processed')
    parser.add_argument('--att_out', action='store_true', help='exterior attention around mlp')
    args = parser.parse_args()

    # dimensions
    config.dim_ = args.dim
    config.motion.dim = args.dim
    config.core.hidden_dim = args.hidden_dim
    config.motion_fc_in.in_features = args.dim
    config.motion_fc_in.out_features = args.hidden_dim
    config.motion_fc_out.in_features = args.hidden_dim
    config.motion_fc_out.out_features = args.dim
    config.att_out = args.att_out
    min_frame_num = args.min_frame
    max_frame_num = args.max_frame

    model = Model(config)
    model.load_state_dict(torch.load(args.model_pth))
    model.eval()
    model.cuda()

    config.motion.amass_target_length = config.motion.amass_target_length_eval
    dataset = AMASSEval(config, 'test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    dct_m, idct_m = get_dct_matrix(config.motion.amass_input_length_dct)  # assuming a suitable input length is set in config
    motion_in_poses, motion_out_poses = regress_pred(model, dataloader, min_frame_num=min_frame_num,
                                                     max_frame_num=max_frame_num, dct_m= dct_m, idct_m=idct_m)

    # convert joint positions to JSON format
    joint_data = {"input_joint_positions": motion_in_poses, "output_joint_positions": motion_out_poses}
    with open('joint_positions.json', 'w') as f:
        json.dump(joint_data, f)

    print('Joint positions saved to joint_positions.json.')

if __name__ == "__main__":
    main()
