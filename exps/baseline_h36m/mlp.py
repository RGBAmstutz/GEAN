import torch
from mpmath import degree
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
# KANS
from config import config
from ChebyKANLayer import ChebyKANLayer as kan
#from BKANLayer import BKANLayer as kan
# imports used for KAT stability
# from https://github.com/Adamdad/rational_kat_cu
#from kat_rational import KAT_Group
#from timm.models.layers import to_2tuple
#from functools import partial

# Implementation of KAN from KAT (Kolmogorov Arnold Transformer : https://github.com/Adamdad/kat)
# act_layer used in KAT: also GELU
#class KAN(nn.Module):
#    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
#    """
#    def __init__(
#            self,
#            in_features,
#            hidden_features=None,
#            out_features=None,
#            act_layer=KAT_Group,
#            norm_layer=None,
#            bias=True,
#            drop=0.,
#            use_conv=False,
#            act_init="gelu",
#    ):
#        super().__init__()
#        out_features = out_features or in_features
#        hidden_features = hidden_features or in_features
#        bias = to_2tuple(bias)
#        drop_probs = to_2tuple(drop)
#        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
#
#        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
#        self.act1 = KAT_Group(mode="identity")
#        self.drop1 = nn.Dropout(drop_probs[0])
#        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
#        self.act2 = KAT_Group(mode=act_init)
#        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
#        self.drop2 = nn.Dropout(drop_probs[1])
#
#        self.reset_parameters()
#
#    def reset_parameters(self):
#        # fc1
#        nn.init.xavier_uniform_(self.fc1.weight, gain=1e-8)  # weight scale (gain)
#        nn.init.constant_(self.fc1.bias, 0)
#        # fc2
#        nn.init.xavier_uniform_(self.fc2.weight, gain=1e-8)  # weight scale (gain)
#        nn.init.constant_(self.fc2.bias, 0)
#
#
#    def forward(self, x):
#        x = self.act1(x)
#        x = self.drop1(x)
#        x = self.fc1(x)
#        x = self.act2(x)
#        x = self.drop2(x)
#        x = self.fc2(x)
#        return x
# END OF KANs from KAT IMPLEMENTATION

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

        # kan --
        self.dim = dim
        self.seq = seq
        # kan --

        #self.reset_parameters()

    # augmented
    #def reset_parameters(self):
    #    # swiglu
    #     nn.init.xavier_uniform_(self.swiglu.swiglu.linear.weight, gain=1e-8)  # weight scale (gain)
    #     nn.init.constant_(self.swiglu.swiglu.linear.bias, 0)
    #    # kan


    def forward(self, x):
        # swiglu implementation
        # ------
        v = self.norm0(x)
        v = self.swiglu(v)
        # kan --
        #v = v.reshape(-1, self.dim, self.seq)
        # kan --
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

#def fourier_2d_pe(seq_len, feat_len, num_freqs=4, device="cpu"):
#    pos_t = torch.arange(seq_len, device=device).unsqueeze(1)  # [seq_len, 1]
#    pos_f = torch.arange(feat_len, device=device).unsqueeze(0) # [1, feat_len]
#
#    pe = torch.zeros(seq_len, feat_len, device=device)
#    for k in range(1, num_freqs + 1):
#        pe += torch.sin(k * pos_t / seq_len) + torch.cos(k * pos_f / feat_len)
#    return pe
#
#def fourier_positional_encoding(seq_len: int, dim: int, num_freqs: int = 8, device="cpu"):
#    """
#    Create Fourier-style sinusoidal encodings [seq_len, dim * 2 * num_freqs]
#    """
#    position = torch.arange(seq_len, device=device).unsqueeze(-1)  # [seq_len, 1]
#    freqs = torch.arange(1, num_freqs + 1, device=device).unsqueeze(0)  # [1, num_freqs]
#
#    # broadcast: [seq_len, num_freqs]
#    angles = position * freqs
#
#    sin_part = torch.sin(angles)  # [seq_len, num_freqs]
#    cos_part = torch.cos(angles)  # [seq_len, num_freqs]
#
#    encoding = torch.cat([sin_part, cos_part], dim=-1)  # [seq_len, 2*num_freqs]
#
#    # project to desired dim
#    proj = torch.randn(2 * num_freqs, dim, device=device)
#    encoding = encoding @ proj  # [seq_len, dim]
#    return encoding
#
#def positional_encoding(seq_len: int, dim: int, device="cpu"):
#    """
#    Create sinusoidal positional encodings of shape [seq_len, dim]
#    """
#    position = torch.arange(seq_len, device=device).unsqueeze(1)  # [seq_len, 1]
#    div_term = torch.exp(
#        torch.arange(0, dim, 2, device=device).float() * -(torch.log(torch.tensor(10000.0)) / dim)
#    )  # [dim/2]
#
#    pe = torch.zeros(seq_len, dim, device=device)
#    pe[:, 0::2] = torch.sin(position * div_term)
#    pe[:, 1::2] = torch.cos(position * div_term)
#    return pe  # [seq_len, dim]

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
            nn.Dropout(dropout)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ff[0].weight, gain=1e-8)  # weight scale (gain)
        nn.init.constant_(self.ff[0].bias, 0)
        nn.init.xavier_uniform_(self.ff[2].weight, gain=1e-8)  # weight scale (gain)
        nn.init.constant_(self.ff[2].bias, 0)

    def forward(self, x):
        # x: [B, T, D]
        # attention (residual)
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out

        # feed-forward (residual)
        x = x + self.ff(self.norm2(x))
        return x

class GatingUnit(nn.Module):
    def __init__(self, in_features, out_features):
        super(GatingUnit, self).__init__()
        # kan --
        #self.kan = kan(in_features, out_features * 2, degree=3)
        #self.kan = KAN(in_features=in_features, out_features=out_features*2)
        # kan --
        if config.fft:
            self.linear = nn.Linear(in_features, out_features * 2, dtype=torch.cfloat) #cfloat for fft  # *2 for splitting into two parts for gating
        else:
            #self.down = nn.Linear(in_features, out_features // 3)
            #self.up = nn.Linear(out_features // 3, out_features)

            #self.up1 = nn.Linear(in_features, out_features *3)
            #self.down1 = nn.Linear(out_features * 3, out_features)
            #self.down2 = nn.Linear(out_features, out_features // 3)
            #self.up2 = nn.Linear(out_features // 3, out_features)
            #self.down1 = nn.Linear(in_features, out_features // 2)
            #self.down2 = nn.Linear(out_features // 2, out_features // 3)
            ##self.down3 = nn.Linear(out_features // 3, out_features // 11)
            #self.up1 = nn.Linear(out_features // 3, out_features // 2)
            #self.up2 = nn.Linear(out_features // 2, out_features)
            ##self.up3 = nn.Linear(out_features // 2, out_features)


            #self.down = kan(in_features, out_features // 3, degree=4)
            #self.up = kan(out_features // 3, out_features, degree=4)
            #self.kanx = kan(config.motion.dim // 3, config.motion.dim // 3, degree=4)
            #self.kany = kan(config.motion.dim // 3, config.motion.dim // 3, degree=4)
            #self.kanz = kan(config.motion.dim // 3, config.motion.dim // 3, degree=4)
            #self.linx = nn.Linear(config.motion.dim // 3, config.motion.dim // 3)
            #self.liny = nn.Linear(config.motion.dim // 3, config.motion.dim // 3)
            #self.linz = nn.Linear(config.motion.dim // 3, config.motion.dim // 3)
            #self.linear = nn.Linear(in_features, out_features)
            #self.linear = nn.Linear(7, 7)
            #self.w1 = nn.Parameter(torch.randn(in_features, out_features))
            #self.w2 = nn.Parameter(torch.randn(in_features, out_features))
            #self.lin0 = nn.Linear(in_features//5, out_features//5)
            #self.lin1 = nn.Linear(in_features // 5, out_features // 5)
            #self.lin2 = nn.Linear(in_features // 5, out_features // 5)
            #self.lin3 = nn.Linear(in_features // 5, out_features // 5)
            #self.lin4 = nn.Linear(in_features // 5, out_features // 5)

            #self.lstm = nn.LSTM(input_size=in_features,hidden_size=32,num_layers=1,batch_first=True)
            #self.lstm_proj = nn.Linear(32, in_features)
            #self.h_n_proj = nn.Linear(32, config.motion.dim)
            #self.c_n_proj = nn.Linear(32, config.motion.dim)


            self.linear = nn.Linear(in_features, out_features * 2)

        if config.att_in:
            self.att = MiniTrans(dim=config.motion.h36m_input_length)

        #self.pe = positional_encoding(config.motion.h36m_input_length, config.motion.dim, device='cuda')
        #self.pe = fourier_positional_encoding(config.motion.h36m_input_length, config.motion.dim, device='cuda')
        #self.pe = fourier_2d_pe(config.motion.h36m_input_length, config.motion.dim, num_freqs=4, device='cuda')
        #self.tsdecomp = TemporalSpatialDecomp(configs=config)
        #self.w1 = nn.Parameter(torch.randn(config.kanfusemlp.h_dim, config.motion.dim))
        #self.w2 = nn.Parameter(torch.randn(config.kanfusemlp.h_dim, config.motion.h36m_input_length))

        self.beta = nn.Parameter(torch.ones(1))
        #self.alpha = nn.Parameter(torch.ones(1))
        #self.gamma = nn.Parameter(torch.ones(1))
        self.reset_parameters()

    def reset_parameters(self):
        # swiglu
        nn.init.xavier_uniform_(self.linear.weight, gain=1e-8)  # weight scale (gain)
        nn.init.constant_(self.linear.bias, 0)
        # harmonic projection
        #nn.init.xavier_uniform_(self.harm_proj.weight, gain=1e-8)
        #nn.init.constant_(self.harm_proj.bias, 0)
        # down/up
        #nn.init.xavier_uniform_(self.up1.weight, gain = 1e-8)
        #nn.init.constant_(self.up1.bias, 0)
        #nn.init.xavier_uniform_(self.up2.weight, gain = 1e-8)
        #nn.init.constant_(self.up2.bias, 0)
        ###nn.init.xavier_uniform_(self.up3.weight, gain = 1e-8)
        ###nn.init.constant_(self.up3.bias, 0)
        #nn.init.xavier_uniform_(self.down1.weight, gain = 1e-8)
        #nn.init.constant_(self.down1.bias, 0)
        #nn.init.xavier_uniform_(self.down2.weight, gain = 1e-8)
        #nn.init.constant_(self.down2.bias, 0)
        ###nn.init.xavier_uniform_(self.down3.weight, gain = 1e-8)
        ###nn.init.constant_(self.down3.bias, 0)
        # spatial splits
        #nn.init.xavier_uniform_(self.linx.weight, gain=1e-8)  # weight scale (gain)
        #nn.init.constant_(self.linx.bias, 0)
        #nn.init.xavier_uniform_(self.liny.weight, gain=1e-8)  # weight scale (gain)
        #nn.init.constant_(self.liny.bias, 0)
        #nn.init.xavier_uniform_(self.linz.weight, gain=1e-8)  # weight scale (gain)
        #nn.init.constant_(self.linz.bias, 0)
        # temporal splits
        #nn.init.xavier_uniform_(self.lin0.weight, gain=1e-8)  # weight scale (gain)
        #nn.init.constant_(self.lin0.bias, 0)
        #nn.init.xavier_uniform_(self.lin1.weight, gain=1e-8)  # weight scale (gain)
        #nn.init.constant_(self.lin1.bias, 0)
        #nn.init.xavier_uniform_(self.lin2.weight, gain=1e-8)  # weight scale (gain)
        #nn.init.constant_(self.lin2.bias, 0)
        #nn.init.xavier_uniform_(self.lin3.weight, gain=1e-8)  # weight scale (gain)
        #nn.init.constant_(self.lin3.bias, 0)
        #nn.init.xavier_uniform_(self.lin4.weight, gain=1e-8)  # weight scale (gain)
        #nn.init.constant_(self.lin4.bias, 0)

    def forward(self, x):
        # kan --
        #v = self.kan(x)
        # kan --
        #v = self.linear(x)
        #x = x.permute(0,2,1)
        #x = x + self.pe.unsqueeze(0).permute(0, 2, 1)
        #x = self.linear(x)
        #temporal, spatial = self.tsdecomp(x)
        #v, l = x.chunk(2, dim=-1)  # split the tensor into two for the gating mechanism
        #gate = F.sigmoid(v@self.w1 + l@self.w2 + self.beta)
        #var = gate*v + (1-gate)*l
        #print(temporal.shape, self.w1.shape, spatial.shape, self.w2.shape)
        #gate = F.sigmoid((temporal@self.w1).permute(0, 2, 1) + spatial@self.w2 + self.beta)
        #gate = torch.sin((temporal@self.w1).permute(0, 2, 1) + spatial@self.w2 + self.beta)
        #gate = (torch.sin((temporal@self.w1).permute(0, 2, 1)))**2 + (torch.cos(spatial@self.w2))**2 + self.beta
        #var = gate*(temporal@self.w1).permute(0,2,1) + (1-gate)*(spatial@self.w2)
        #return var
        #print(f"before lin: {x.shape}")
        #print(f"after lin: {lin.shape}")
        #v, l, n = lin.chunk(3, dim=-1)  # split the tensor into two for the gating mechanism
        #v, l, n, p = lin.chunk(4, dim=-1)  # split the tensor into two for the gating mechanism
        #gate = v * F.sigmoid(l + n) * p * self.beta
        #gate = v * l * self.beta
        #gate = F.sigmoid(gate)

        #linear = self.linear(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        #return linear

        #down = self.down(x)
        #down = F.sigmoid(down)
        ##up = self.up(down)

        #down1 = self.down1(x)
        #down2 = self.down2(down1)
        ##down3 = self.down3(down2)
        #up1 = self.up1(down2)
        #up2 = self.up2(up1)
        ##up3 = self.up3(up2)

        #return linear + up
        #return gate
        #return linear + up2

        #x = x.permute(0, 2, 1)
        #B, T, S = x.shape
        #axis_split = x.reshape(B, T, 22, 3)
        #x_ = self.linx(axis_split[..., 0])
        #y_ = self.liny(axis_split[..., 1])
        #z_ = self.linz(axis_split[..., 2])
        #spatial_concat = torch.concat((x_, y_, z_), dim=-1).permute(0, 2, 1)
        ##return concat #F.sigmoid(concat)
        #x = x.permute(0, 2, 1)
        #B , S, T =x.shape
        #temporal_split = x.reshape(B, S, T//5, 5)
        #_0 = self.lin0(temporal_split[..., 0])
        #_1 = self.lin1(temporal_split[..., 1])
        #_2 = self.lin2(temporal_split[..., 2])
        #_3 = self.lin3(temporal_split[..., 3])
        #_4 = self.lin4(temporal_split[..., 4])
        #temporal_concat = torch.concat((_0, _1, _2, _3, _4), dim=-1)
        #return concat
        #return linear + temporal_concat

        #lstm, mem = self.lstm(x)
        #h_n, c_n = mem
        #h_n_seq = h_n.squeeze(0).unsqueeze(-1).repeat(1,1,50)
        #c_n_seq = c_n.squeeze(0).unsqueeze(-1).repeat(1,1,50)
        #prev = self.h_n_proj(h_n_seq.permute(0, 2, 1)).permute(0, 2, 1)
        #long = self.c_n_proj(c_n_seq.permute(0, 2, 1)).permute(0, 2, 1)
        #lstm_proj = self.lstm_proj(lstm)
        #lin = self.linear(x * F.sigmoid(lstm_proj))
        #v, l = lin.chunk(2, dim=-1)
        #gmlp =  v * F.sigmoid(l) * self.beta # apply sigmoid activation
        #return gmlp #+ lstm_proj

        #B, S, T = x.shape
        #J = S // 3  # number of joints

        ## Reshape [B, T, S] → [B*T, J, 3]
        #harm = x.reshape(B * T, J, 3)
        #harm = sh_expand_batch(harm, deg=self.deg).reshape(B, J, self.C, T)

        #lin = self.linear(x)
        #v, l = lin.chunk(2, dim=-1)
        #gate = v * F.sigmoid(l) * self.beta # apply sigmoid activation
        #return gate
        #return self.harm_proj(harm.reshape(B, J, T, self.C)).reshape(B, S, T) + gate

        lin = self.linear(x)
        v, l = lin.chunk(2, dim=-1)
        gate = v * F.sigmoid(l) * self.beta # apply sigmoid activation
        return gate
        #return gate + self.att(x)
        #return self.att(gate)
        #return (self.alpha) * gate + (self.alpha - 1) * self.att(x)

        #lin = self.linear(x)
        #v, l = lin.chunk(2, dim=-1)
        #return v * F.sigmoid(l) * self.beta # apply sigmoid activation

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
        nn.init.xavier_uniform_(self.spatial_fc.weight, gain=1e-8)  # weight scale (gain)
        nn.init.constant_(self.spatial_fc.bias, 0)
        nn.init.xavier_uniform_(self.temporal_fc.weight, gain=1e-8)  # weight scale (gain)
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

    #def forward(self, x):
    #    B, T, S, E = x.shape
    #    if self.mode == 'linear':
    #        # spatial component
    #        spatial = x.permute(0, 2, 1, 3).contiguous()
    #        spatial = spatial.view(B*S, E, T)
    #        spatial = self.spatial_fc(spatial)
    #        spatial_proj = spatial.view(B, E, S, self.configs.kanfusemlp.h_dim)

    #        # temporal component
    #        temporal = x.reshape(B*T, E, S).contiguous()
    #        temporal = self.temporal_fc(temporal)
    #        temporal_proj = temporal.view(B, E, T, self.configs.kanfusemlp.h_dim)

    #    else: # mean
    #        spatial = x.mean(dim=1, keepdim=True)
    #        spatial = spatial.permute(0, 2, 1, 3)
    #        spatial_proj = spatial.expand(-1, -1, -1, self.configs.kanfuse.h_dim)

    #        temporal = x.mean(dim=2, keepdim=True)
    #        temporal_proj = temporal.expand(-1, -1, -1, self.configs.kanfuse.h_dim)

    #    return temporal_proj, spatial_proj

# PyTorch-friendly utilities: Cartesian → Spherical coordinates → Real spherical harmonics (up to l=2)
# These are modular, GPU-compatible functions you can drop into a training pipeline.
# They compute *real* spherical harmonics as commonly used in graphics / SH lighting (Peter-Pike Sloan style).
# The implementation returns SH coefficients for degrees l=0,1,2 (total 1 + 3 + 5 = 9 channels).
#
# Functions:
#  - cartesian_to_spherical(xyz) -> (r, theta, phi)
#  - sh_basis_l0_l2(xyz) -> tensor(..., 9)  # real SH basis values for each 3D point, normalized to unit radius
#  - sh_expand_batch(xyz_batch, deg=2) -> [B, J, num_coeffs] for batch of joints
#
# Example use:
#   xyz = torch.randn(2, 22, 3)  # B, J, 3
#   coeffs = sh_expand_batch(xyz, deg=2)  # [2, 22, 9]
#

from typing import Tuple

def cartesian_to_spherical(xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert cartesian coordinates to spherical coordinates.
    Input: xyz tensor of shape (..., 3) where last dim is (x,y,z).
    Output:
      r     : (...,) radius (>=0)
      theta : (...,) polar angle (colatitude) in [0, pi] (angle from +z axis)
      phi   : (...,) azimuth angle in [-pi, pi] (angle from +x in x-y plane)
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    r = torch.sqrt(x*x + y*y + z*z + 1e-12)
    # avoid division by zero
    theta = torch.acos(torch.clamp(z / r, -1.0, 1.0))
    phi = torch.atan2(y, x)
    return r, theta, phi

def sh_basis_l0_l2(xyz: torch.Tensor) -> torch.Tensor:
    """
    Compute real spherical harmonics basis functions up to degree l=2 for points xyz.
    Input: xyz (..., 3) or (B, J, 3)
    Output: (..., 9) channels in the following order:
      [Y00,
       Y1-1, Y10, Y11,
       Y2-2, Y2-1, Y20, Y21, Y22]
    These are *real* SH basis values commonly used in graphics (Sloan et al.).
    The basis expects input directions on the unit sphere; the function will normalize internally.
    """
    orig_shape = xyz.shape
    assert orig_shape[-1] == 3, "last dim must be 3 (x,y,z)"
    # Flatten to (...,3) for convenience
    flat = xyz.reshape(-1, 3)
    x = flat[:, 0]
    y = flat[:, 1]
    z = flat[:, 2]
    # Normalize to unit vectors (if r==0, replace with small epsilon)
    r = torch.sqrt(x*x + y*y + z*z + 1e-12)
    xn = x / r
    yn = y / r
    zn = z / r

    # Precomputed constants (real SH normalization constants)
    c0 = 0.28209479177387814        # Y00 = 1/2 * sqrt(1/pi)
    c1 = 0.4886025119029199         # sqrt(3/(4pi))
    c2 = 1.0925484305920792         # sqrt(15/(4pi))
    c3 = 0.31539156525252005        # sqrt(5/(16pi))
    c4 = 0.5462742152960396         # sqrt(15/(16pi))

    # l=0
    Y00 = torch.full_like(xn, c0)

    # l=1
    Y1_m1 = c1 * yn   # Y1,-1  ~ y
    Y1_0  = c1 * zn   # Y1,0   ~ z
    Y1_1  = c1 * xn   # Y1,1   ~ x

    # l=2 (five terms)
    Y2_m2 = c2 * (xn * yn)                 # ~ xy
    Y2_m1 = c2 * (yn * zn)                 # ~ yz
    Y2_0  = c3 * (3.0 * zn * zn - 1.0)     # ~ (3z^2 - 1)
    Y2_1  = c2 * (xn * zn)                 # ~ xz
    Y2_2  = c4 * (xn * xn - yn * yn)       # ~ (x^2 - y^2)

    out = torch.stack([Y00,
                       Y1_m1, Y1_0, Y1_1,
                       Y2_m2, Y2_m1, Y2_0, Y2_1, Y2_2], dim=-1)
    out = out.reshape(*orig_shape[:-1], 9)
    return out

def sh_expand_batch(xyz_batch: torch.Tensor, deg: int = 2) -> torch.Tensor:
    """
    Expand a batch of joint 3D positions into real spherical harmonic coefficients up to degree `deg`.
    Supports deg up to 2 in this helper. Returns shape [B, J, C] where C = (deg+1)^2.
    Input: xyz_batch [B, J, 3] or [B, 3, J] (spatial dims last recommended). We'll accept [B, J, 3].
    """
    assert deg in (0, 1, 2), "This helper supports deg=0,1,2"
    B, J, D = xyz_batch.shape
    assert D == 3, "Expected last dim = 3"
    if deg == 0:
        # only Y00
        out = torch.full((B, J, 1), 0.28209479177387814, dtype=xyz_batch.dtype, device=xyz_batch.device)
        return out
    elif deg == 1:
        # 1 + 3 = 4 channels
        flat = xyz_batch.reshape(-1, 3)
        sh = sh_basis_l0_l2(flat).reshape(B, J, 9)  # compute up to l=2 then slice
        return sh[..., :4]
    else:
        # deg==2 -> 9 channels
        return sh_basis_l0_l2(xyz_batch)