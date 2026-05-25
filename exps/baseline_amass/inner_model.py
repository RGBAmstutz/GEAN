import copy

from torch import cfloat
from torch import nn
from SphericalMultiKAN import build_model
from einops.layers.torch import Rearrange
# KANS
from config import config
# attention
import torch
from torch.nn import functional as F

class MiniTrans(nn.Module):
    def __init__(self, dim, num_heads=2, ff_mult=2, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Linear(ff_mult * dim, dim),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ff[0].weight, gain=1e-11)  # weight scale (gain)
        nn.init.constant_(self.ff[0].bias, 0)
        nn.init.xavier_uniform_(self.ff[2].weight, gain=1e-11)  # weight scale (gain)
        nn.init.constant_(self.ff[2].bias, 0)

    def forward(self, x):
        # x: [B, T, D]
        # attention (residual)
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out

        # feed-forward (residual)
        x = x + self.ff(self.norm2(x))
        return x

class EfficientAttention(nn.Module):
    def __init__(self, in_channels=1, key_channels=32, value_channels=32, head_count=8):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.head_count = head_count

        assert key_channels % head_count == 0
        assert value_channels % head_count == 0

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, x):
        n, _, h, w = x.size()
        keys = self.keys(x).reshape(n, self.head_count, self.key_channels // self.head_count, h * w)
        queries = self.queries(x).reshape(n, self.head_count, self.key_channels // self.head_count, h * w)
        values = self.values(x).reshape(n, self.head_count, self.value_channels // self.head_count, h * w)

        keys = F.softmax(keys, dim=-1)
        queries = F.softmax(queries, dim=-2)

        context = torch.einsum("bhcn,bhvn->bhcv", keys, values)
        attended = torch.einsum("bhvc,bhcn->bhvn", context, queries)
        attended = attended.reshape(n, self.value_channels, h, w)

        out = self.reprojection(attended)
        return out + x

class MLP(nn.Module):
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        super(MLP, self).__init__()
        seq = self.config.core.seq_len
        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

        self.core = build_model(self.config.core)

        self.temporal_fc_in = config.motion_fc_in.temporal_fc
        self.temporal_fc_out = config.motion_fc_out.temporal_fc
        self.J = config.motion.dim // 3
        if self.temporal_fc_in:
            if config.fft:
                self.motion_fc_in = nn.Linear(self.config.motion.amass_input_length_dct, self.config.motion.amass_input_length_dct, dtype=cfloat)
            else:
                if config.conv:
                    self.conv = nn.Conv1d(self.config.motion.amass_input_length_dct, self.config.motion.amass_input_length_dct, kernel_size=config.kernel_size, padding=config.pad)
                if config.att_out:
                    self.att_in = EfficientAttention(in_channels=self.config.motion.dim)
                self.motion_fc_in = nn.Linear(self.config.motion.amass_input_length_dct, self.config.motion.amass_input_length_dct)
        else:
            if config.fft:
                self.motion_fc_in = nn.Linear(self.config.motion.dim, self.config.motion.dim, dtype=cfloat)
            else:
                if config.conv:
                    self.conv = nn.Conv1d(self.config.motion.amass_input_length, self.config.motion.amass_input_length, kernel_size=config.kernel_size, padding=config.pad)
                if config.att_out:
                    self.att_in = EfficientAttention(in_channels=1)
                self.motion_fc_in = nn.Linear(self.config.motion.dim, self.config.motion.dim)

        if self.temporal_fc_out:
            if config.ifft:
                self.motion_fc_out = nn.Linear(self.config.motion.amass_input_length_dct, self.config.motion.amass_input_length_dct, dtype=cfloat)
            else: 
                self.motion_fc_out = nn.Linear(self.config.motion.amass_input_length_dct, self.config.motion.amass_input_length_dct)
            if config.att_out:
                self.att_out = EfficientAttention(in_channels=1)
        else:
            if config.ifft:
                self.motion_fc_out = nn.Linear(self.config.motion.dim, self.config.motion.dim, dtype=cfloat)
            else:
                self.motion_fc_out = nn.Linear(self.config.motion.dim, self.config.motion.dim)
            if config.att_out:
                self.att_out = EfficientAttention(in_channels=1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def forward(self, motion_input):
        if config.conv:
            motion_input = self.conv(motion_input)
        # fft formatting
        if config.fft:
            motion_input = motion_input.cfloat()

        if self.temporal_fc_in:
            motion_feats_shaped = self.arr0(motion_input)
            motion_feats = self.motion_fc_in(motion_feats_shaped)
        else:
            motion_feats = self.motion_fc_in(motion_input)
            motion_feats = self.arr0(motion_feats)

        if config.att_out:
            motion_feats = motion_feats.unsqueeze(1)
            motion_feats = self.att_in(motion_feats)
            motion_feats = motion_feats.squeeze()
            att_in = motion_feats

        motion_feats_core = self.core(motion_feats)

        if config.att_out:
            motion_feats_core = motion_feats_core.unsqueeze(1)
            motion_feats_core = self.att_out(motion_feats_core)
            motion_feats_core = motion_feats_core.squeeze()
            motion_feats_core += att_in

        if self.temporal_fc_out:
            motion_feats = self.motion_fc_out(motion_feats_core)
            motion_feats = self.arr1(motion_feats)
        else:
            motion_feats_shaped = self.arr1(motion_feats_core)
            motion_feats = self.motion_fc_out(motion_feats_shaped)

        return motion_feats
