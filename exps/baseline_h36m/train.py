import argparse
import json
import copy

from outer_model import outer_model
from config import config
from inner_model import MLP as Model

# learning rate scheduler
import torch.optim.lr_scheduler as lr_scheduler
#########
from datasets.h36m import H36MDataset
from utils.logger import get_logger, print_and_log_info
from utils.pyt_utils import link_file, ensure_dir
from datasets.h36m_eval import H36MEval

from test import test

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp-name', type=str, default=None, help='=exp name')
parser.add_argument('--seed', type=int, default=888, help='=seed')
parser.add_argument('--temporal-only', action='store_true', help='=temporal only')
parser.add_argument('--layer-norm-axis', type=str, default='spatial', help='=layernorm axis')
parser.add_argument('--with-normalization', action='store_true', help='=use layernorm')
parser.add_argument('--spatial-fc', action='store_true', help='=use only spatial fc')
parser.add_argument('--num', type=int, default=64, help='=num of blocks')
parser.add_argument('--weight', type=float, default=1., help='=loss weight')
parser.add_argument('--more-ckpt-info', action='store_true', help='=more checkpoint info in pth file')
parser.add_argument('--dim', type=int, default=66, help='=number of dimensions')
parser.add_argument('--hidden-dim', type=int, default=66, help='=number of hidden dimensions')
parser.add_argument('--dct', action='store_true', help='=enable discrete cosine transform')
parser.add_argument('--ffm', action='store_true', help='=enable fourier feature map')
parser.add_argument('--fft', action='store_true', help='=enable fourier transform')

parser.add_argument('--pool', type=str, default=None, help='=type of pooling: {max, avg}')
parser.add_argument('--conv', action='store_true', help='=convolution')
parser.add_argument('--att_out', action='store_true', help='=outer attention')
parser.add_argument('--att_in', action='store_true', help='=inner attention')

# learning rate scheduler
parser.add_argument('--lr_scheduler', type=str, default=None, help='=type of lr scheduler: {full, half, 30k}')

# PatchMLP parser arguments
parser.add_argument('--seq_len', type=int, help='=patchMLP sequence length')
parser.add_argument('--pred_len', type=int, help='=patchMLP predict length')
parser.add_argument('--use_norm', action='store_true', help='=patchMLP normalization')
parser.add_argument('--d_model', type=int, help='=patchMLP model dimension')
parser.add_argument('--e_layers', type=int, help='=patchMLP number of encoder layers')
parser.add_argument('--enc_in', type=int, help='=patchMLP encoder input size')

# harmonic pre/postprocessing
parser.add_argument('--harm', action='store_true', help='=harmonic pre/postprocessing')

args = parser.parse_args()

torch.use_deterministic_algorithms(True)
acc_log = open(args.exp_name, 'a')
torch.manual_seed(args.seed)
writer = SummaryWriter()

config.motion_fc_in.temporal_fc = args.temporal_only
config.motion_fc_out.temporal_fc = args.temporal_only
config.core.norm_axis = args.layer_norm_axis
config.core.spatial_fc_only = args.spatial_fc
config.core.with_normalization = args.with_normalization
config.core.num_layers = args.num
config.pre_dct = args.dct
config.post_dct = args.dct
config.deriv_input = args.dct
config.deriv_output = args.dct

# fft
config.fft = args.fft
config.ifft = args.fft
config.ffm = args.ffm

# outer model
config.conv = args.conv
config.pool = args.pool
config.att_out = args.att_out
config.att_in = args.att_in

# harmonic pre/postprocessing
config.harm = args.harm

# dimensions
config.dim_ = args.dim
config.motion.dim = args.dim
config.core.hidden_dim = args.hidden_dim
config.motion_fc_in.in_features = args.dim
config.motion_fc_in.out_features = args.hidden_dim
config.motion_fc_out.in_features = args.hidden_dim
config.motion_fc_out.out_features = args.dim

acc_log.write(''.join('Seed : ' + str(args.seed) + '\n'))

def update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer) :
    if nb_iter > 30000:
        current_lr = 1e-5
    else:
        current_lr = 3e-4

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

# for scheduled learning rate
def update_lr_scheduler(nb_iter, total_iter, optimizer, scheduler) :
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

def gen_velocity(m):
    dm = m[:, 1:] - m[:, :-1]
    return dm

def train_step(h36m_motion_input, h36m_motion_target, model, optimizer, nb_iter, total_iter, max_lr, min_lr, scheduler) :

    motion_pred = outer_model(h36m_motion_input, model)

    b,n,c = h36m_motion_target.shape
    motion_pred = motion_pred.reshape(b, n, 22, 3).reshape(-1, 3)
    h36m_motion_target = h36m_motion_target.cuda().reshape(b, n, 22, 3).reshape(-1, 3)
    loss = torch.mean(torch.norm(motion_pred - h36m_motion_target, 2, 1))

    if config.use_relative_loss:
        motion_pred = motion_pred.reshape(b,n,22,3)
        dmotion_pred = gen_velocity(motion_pred)
        motion_gt = h36m_motion_target.reshape(b,n,22,3)
        dmotion_gt = gen_velocity(motion_gt)
        dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1, 3), 2, 1))
        loss = loss + dloss
    else:
        loss = loss.mean()

    writer.add_scalar('Loss/angle', loss.detach().cpu().numpy(), nb_iter)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler:
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
    else:
        optimizer, current_lr = update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer)

    writer.add_scalar('LR/train', current_lr, nb_iter)
    return loss.item(), optimizer, current_lr

model = Model(config)
model.train()
model.cuda()

config.motion.h36m_target_length = config.motion.h36m_target_length_train
dataset = H36MDataset(config, 'train', config.data_aug)

shuffle = True
sampler = None
dataloader = DataLoader(dataset, batch_size=config.batch_size,
                        num_workers=config.num_workers, drop_last=True,
                        sampler=sampler, shuffle=shuffle, pin_memory=True)

eval_config = copy.deepcopy(config)
eval_config.motion.h36m_target_length = eval_config.motion.h36m_target_length_eval
eval_dataset = H36MEval(eval_config, 'test')

shuffle = False
sampler = None
eval_dataloader = DataLoader(eval_dataset, batch_size=128,
                        num_workers=1, drop_last=False,
                        sampler=sampler, shuffle=shuffle, pin_memory=True)


# initialize optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=config.cos_lr_max, 
                             weight_decay=config.weight_decay)

# define lambda that decreases over the span of the specified min and max lr
def lr_lambda_full_decay(epoch):
    decay = min(epoch / config.cos_lr_total_iters, 1)
    scale = 1.0 - decay
    return scale * ((config.cos_lr_max - config.cos_lr_min) / config.cos_lr_max) + (config.cos_lr_min / config.cos_lr_max)


# define lambda that decreases after halfway point linearly interpolated between min and max lr
def lr_lambda_halfpoint_decay(epoch):
    if epoch < int(config.cos_lr_total_iters / 2):
        return 1
    progress = (epoch - int(config.cos_lr_total_iters / 2)) / int(config.cos_lr_total_iters / 2)
    decay = min(progress, 1)
    scale = 1.0 - decay
    return scale * ((config.cos_lr_max - config.cos_lr_min) / config.cos_lr_max) + (config.cos_lr_min / config.cos_lr_max)

def lr_lambda_30k_decay(epoch):
    if epoch < 30000:
        return 1
    progress = (epoch - 30000) / 30000
    decay = min(progress, 1)
    scale = 1.0 - decay
    return scale * ((config.cos_lr_max - config.cos_lr_min) / config.cos_lr_max) + (config.cos_lr_min / config.cos_lr_max)

scheduler = None
if args.lr_scheduler == "full":
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_full_decay)
elif args.lr_scheduler == "half":
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_halfpoint_decay)
elif args.lr_scheduler == "30k":
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_30k_decay)

ensure_dir(config.snapshot_dir)
logger = get_logger(config.log_file, 'train')
link_file(config.log_file, config.link_log_file)

print_and_log_info(logger, json.dumps(config, indent=4, sort_keys=True))

if config.model_pth is not None :
    state_dict = torch.load(config.model_pth)
    model.load_state_dict(state_dict, strict=True)
    print_and_log_info(logger, "Loading model path from {} ".format(config.model_pth))

##### ------ training ------- #####
nb_iter = 0
avg_loss = 0.
avg_lr = 0.

while (nb_iter + 1) < config.cos_lr_total_iters:

    for (h36m_motion_input, h36m_motion_target) in dataloader:
        loss, optimizer, current_lr = train_step(h36m_motion_input, h36m_motion_target, model, optimizer, nb_iter, config.cos_lr_total_iters, config.cos_lr_max, config.cos_lr_min, scheduler)
        avg_loss += loss
        avg_lr += current_lr

        if (nb_iter + 1) % config.print_every == 0:
            avg_loss = avg_loss / config.print_every
            avg_lr = avg_lr / config.print_every

            print_and_log_info(logger, "Iter {} Summary: ".format(nb_iter + 1))
            print_and_log_info(logger, f"\t lr: {avg_lr} \t Training loss: {avg_loss}")
            avg_loss = 0
            avg_lr = 0

        if (nb_iter + 1) % config.save_every == 0:
            torch.save({'lr': current_lr, 'step': nb_iter + 1, 'state_dict': model.state_dict(),
                       'optimizer': optimizer.state_dict(), 'error': avg_loss}, config.snapshot_dir + '/model-iter-' + str(nb_iter + 1) + '.pth') \
                if args.more_ckpt_info else torch.save(model.state_dict(), config.snapshot_dir + '/model-iter-' + str(nb_iter + 1) + '.pth')
            model.eval()
            acc_tmp = test(eval_config, model, eval_dataloader)
            print(acc_tmp)
            acc_log.write(''.join(str(nb_iter + 1) + '\n'))
            line = ''
            for ii in acc_tmp:
                line += str(ii) + ' '
            line += '\n'
            acc_log.write(''.join(line))
            model.train()

        if (nb_iter + 1) == config.cos_lr_total_iters :
            break
        nb_iter += 1

writer.close()
