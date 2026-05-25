import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from config import config

class LN(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, dim, 1]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim, 1]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


class LN_v2(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class Spatial_SwiGLU(nn.Module):
    def __init__(self, dim):
        super(Spatial_SwiGLU, self).__init__()
        self.swiglu = GatingUnit(dim, dim)
        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

    def forward(self, x):
        x = self.arr0(x)
        x = self.swiglu(x)
        x = self.arr1(x)
        return x

class Temporal_SwiGLU(nn.Module):
    def __init__(self, dim):
        super(Temporal_SwiGLU, self).__init__()
        self.swiglu = GatingUnit(dim, dim)

    def forward(self, x):
        x = self.swiglu(x)
        return x

class MLP(nn.Module):

    def __init__(self, dim, seq, use_norm=True, use_spatial_fc=False, layernorm_axis='spatial'):
        super().__init__()
        # swiglu implementation
        # --------------------
        if use_spatial_fc:
            self.swiglu = Spatial_SwiGLU(dim)
        else:
            self.swiglu = Temporal_SwiGLU(seq)
        # --------------------

        if use_norm:
            if layernorm_axis == 'spatial':
                self.norm0 = LN(dim)
            elif layernorm_axis == 'temporal':
                self.norm0 = LN_v2(seq)
            elif layernorm_axis == 'all':
                self.norm0 = nn.LayerNorm([dim, seq])
            else:
                raise NotImplementedError
        else:
            self.norm0 = nn.Identity()

    def forward(self, x):
        # swiglu implementation
        # ------
        v = self.norm0(x)
        v = self.swiglu(v)
        return x + v

class TransMLP(nn.Module):
    def __init__(self, dim, seq, use_norm, use_spatial_fc, num_layers, layernorm_axis):
        super().__init__()
        self.mlps = nn.Sequential(*[
            MLP(dim, seq, use_norm, use_spatial_fc, layernorm_axis)
            for i in range(num_layers)])

    def forward(self, x):
        x = self.mlps(x)
        return x

def build_mlps(args):
    if 'seq_len' in args:
        seq_len = args.seq_len
    else:
        seq_len = None
    return TransMLP(
        dim=args.hidden_dim,
        seq=seq_len,
        use_norm=args.with_normalization,
        use_spatial_fc=args.spatial_fc_only,
        num_layers=args.num_layers,
        layernorm_axis=args.norm_axis,
    )

class GatingUnit(nn.Module):
    def __init__(self, in_features, out_features):
        super(GatingUnit, self).__init__()

        if config.fft:
            self.linear = nn.Linear(in_features, out_features * 2, dtype=torch.cfloat) #cfloat for fft  # *2 for splitting into two parts for gating
        else:
            self.linear = nn.Linear(in_features, out_features * 2)

        self.beta = nn.Parameter(torch.ones(1))
        self.reset_parameters()

    def reset_parameters(self):
        # swiglu
        nn.init.xavier_uniform_(self.linear.weight, gain=1e-10)  # weight scale (gain)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        lin = self.linear(x)
        v, l = lin.chunk(2, dim=-1)
        gate = v * F.sigmoid(l) * self.beta # apply sigmoid activation
        return gate

class TemporalSpatialDecomp(nn.Module):
    def __init__(self, configs, mode = 'linear'):
        super(TemporalSpatialDecomp, self).__init__()
        self.configs = configs
        self.mode  = mode
        if mode == 'linear':
            self.spatial_fc = nn.Linear(configs.kanfusemlp.temporal_dim, configs.kanfusemlp.h_dim)
            self.temporal_fc = nn.Linear(configs.kanfusemlp.spatial_dim, configs.kanfusemlp.h_dim)
        elif mode == 'mean':
            pass
        else:
            raise ValueError("mode must be 'linear' or 'mean'")

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.spatial_fc.weight, gain=1e-11)  # weight scale (gain)
        nn.init.constant_(self.spatial_fc.bias, 0)
        nn.init.xavier_uniform_(self.temporal_fc.weight, gain=1e-11)  # weight scale (gain)
        nn.init.constant_(self.temporal_fc.bias, 0)

    def forward(self, x):
        B, T, S = x.shape
        if self.mode == 'linear':
            # spatial component
            spatial = x.permute(0, 2, 1).contiguous()
            spatial = spatial.view(B*S, T)
            spatial = self.spatial_fc(spatial)
            spatial_proj = spatial.view(B, S, self.configs.kanfusemlp.h_dim)

            # temporal component
            temporal = x.reshape(B*T, S).contiguous()
            temporal = self.temporal_fc(temporal)
            temporal_proj = temporal.view(B, T, self.configs.kanfusemlp.h_dim)

        else: # mean
            spatial = x.mean(dim=1, keepdim=True)
            spatial = spatial.permute(0, 2, 1)
            spatial_proj = spatial.expand(-1, -1, self.configs.kanfuse.h_dim)

            temporal = x.mean(dim=2, keepdim=True)
            temporal_proj = temporal.expand(-1, -1, self.configs.kanfuse.h_dim)

        return temporal_proj, spatial_proj
