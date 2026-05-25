import torch
from torch import nn
import numpy as np
import math
import torch.nn.functional as F
# KANS
from config import config
from ChebyKANLayer import ChebyKANLayer as kan
# blocks
from mlp import TransMLP
from timekan.TimeKAN import Model as TK
from mlp import TemporalSpatialDecomp

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=-1)

        x = self.avg(x)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Encoder(nn.Module):

    def __init__(self, d_model, enc_in, embed_dim=None, temporal_spatial=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(enc_in)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff1 = nn.Sequential(
            kan(enc_in, enc_in, embed_dim=embed_dim, degree=3),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.ff2 = nn.Sequential(
            kan(d_model, d_model, embed_dim=embed_dim, degree=3),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.temp_spat = temporal_spatial

    def forward(self, x):
        y_0 = self.ff1(x)
        y_0 = y_0 + x
        y_0 = y_0.permute(0, 3, 1, 2)
        y_1 = self.norm1(y_0)
        if self.temp_spat:
            y_1 = y_1.permute(0, 3, 2, 1)
        else:
            y_1 = y_1.permute(0, 2, 3, 1)
        y_1 = self.ff2(y_1)
        if self.temp_spat:
            y_1 = y_1.permute(0, 3, 2, 1)
        else:
            y_1 = y_1.permute(0, 3, 1, 2)
        x = x.permute(0, 3, 1, 2)
        y_2 = y_1 * y_0 + x
        y_2 = self.norm1(y_2)
        y_2 = y_2.permute(0, 2, 3, 1)

        return y_2

class SphericalMultiKAN(nn.Module):
    def __init__(self, hidden_dim, seq_len, with_normalization, spatial_fc_only, num_layers, norm_axis):
        super().__init__()
        # load
        rng = torch.load('rng_state.pt', weights_only=True)
        torch.set_rng_state(rng['cpu_state'])
        torch.cuda.set_rng_state(rng['cuda_state'])

        # freqmlp
        self.transmlp = TransMLP(
            dim=hidden_dim,
            seq=seq_len,
            use_norm=with_normalization,
            use_spatial_fc=spatial_fc_only,
            num_layers=num_layers,
            layernorm_axis=norm_axis,
        )

    def forward(self, x):
        mlp = self.transmlp(x)
        x = mlp

        return x

def build_model(args):
    if 'seq_len' in args:
        seq_len = args.seq_len
    else:
        seq_len = None

    return SphericalMultiKAN(
        hidden_dim=args.hidden_dim,
        seq_len=seq_len,
        with_normalization=args.with_normalization,
        spatial_fc_only=args.spatial_fc_only,
        num_layers=args.num_layers,
        norm_axis=args.norm_axis,
    )

class SphericalEncoding(nn.Module):
    def __init__(self):
        super(SphericalEncoding, self).__init__()
        self.scale = nn.Parameter(torch.randn(1))
        self.shift = nn.Parameter(torch.randn(1))

    def forward(self, x):
        # normalize between 0 and 1
        x_scaled = (x - x.min()) / (x.max() - x.min())
        # map to unit sphere
        theta = 2 * np.pi * self.scale * x_scaled + self.shift # azimuth
        phi = np.pi * x_scaled # polar

        # sine and cosine for both angles
        sin_phi = torch.sin(phi)
        sin_theta = torch.sin(theta)

        cos_phi = torch.cos(phi)
        cos_theta = torch.cos(theta)

        # create embed
        embedding = torch.stack([sin_phi * cos_theta, sin_phi * sin_theta, cos_phi], dim=-1)

        return embedding

    def decode(self, embeddings):
        # recover angles
        sin_phi_cos_theta = embeddings[..., 0]
        sin_phi_sin_theta = embeddings[..., 1]
        cos_phi = embeddings[..., 2]

        # polar angle
        recovered_phi = torch.acos(cos_phi)
        # azimuth angle
        recovered_theta = torch.atan2(sin_phi_sin_theta, sin_phi_cos_theta)

        # decode back to normalized value
        x_decoded = (recovered_theta - self.shift) / self.scale
        x_decoded = (x_decoded / (2*np.pi))

        return x_decoded, recovered_phi
