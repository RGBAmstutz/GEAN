import torch
import torch.nn as nn
import torch.nn.functional as F
from timekan.layers.Autoformer_EncDec import series_decomp
from timekan.layers.Embed import DataEmbedding_wo_pos
from timekan.layers.StandardNorm import Normalize
from timekan.layers.ChebyKANLayer import ChebyKANLinear
import math

## ARCHITECTURE
# DCT
# EMBEDDING
# TEMPORAL/SPATIAL
# KAN (MLP)
# FUSION
# FC
# FC(?)
# IDCT

class ChebyKANLayer(nn.Module):
    def __init__(self, in_features, out_features,order):
        super().__init__()
        self.fc1 = ChebyKANLinear(
                            in_features,
                            out_features,
                            order)
    def forward(self, x):
        B, N, C = x.shape
        x = self.fc1(x.reshape(B*N,C))
        x = x.reshape(B,N,-1).contiguous()
        return x

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


class FrequencyDecomp(nn.Module):

    def __init__(self, configs):
        super(FrequencyDecomp, self).__init__()
        self.configs = configs

    def forward(self, level_list):
      
        level_list_reverse = level_list.copy()
        level_list_reverse.reverse()
        out_low = level_list_reverse[0]
        out_high = level_list_reverse[1]
        out_level_list = [out_low]
        for i in range(len(level_list_reverse) - 1):
            out_high_res = self.frequency_interpolation(out_low.transpose(1,2),
                                                        self.configs.timekan.seq_len // (self.configs.timekan.down_sampling_window ** (self.configs.timekan.down_sampling_layers-i)),
                                                        self.configs.timekan.seq_len // (self.configs.timekan.down_sampling_window ** (self.configs.timekan.down_sampling_layers-i-1))
                                                        ).transpose(1,2)
            out_high_left = out_high - out_high_res
            out_low = out_high
            if i + 2 <= len(level_list_reverse) - 1:
                out_high = level_list_reverse[i + 2]    
            out_level_list.append(out_high_left) 
        out_level_list.reverse()
        return out_level_list   
    
    def frequency_interpolation(self,x,seq_len,target_len):
        len_ratio = seq_len/target_len
        x_fft = torch.fft.rfft(x, dim=2)
        out_fft = torch.zeros([x_fft.size(0),x_fft.size(1),target_len//2+1],dtype=x_fft.dtype).to(x_fft.device)
        out_fft[:,:,:seq_len//2+1] = x_fft
        out = torch.fft.irfft(out_fft, dim=2)
        out = out * len_ratio
        return out

class FrequencyMixing(nn.Module):

    def __init__(self, configs):
        super(FrequencyMixing, self).__init__()
        self.configs = configs
        self.front_block = M_KAN(configs.timekan.d_model,
                                 self.configs.timekan.seq_len // (self.configs.timekan.down_sampling_window ** (self.configs.timekan.down_sampling_layers)),
                                 order=configs.timekan.begin_order)
                  
          
        self.front_blocks = torch.nn.ModuleList(
                [
                    M_KAN(configs.timekan.d_model,
                          self.configs.timekan.seq_len // (self.configs.timekan.down_sampling_window ** (self.configs.timekan.down_sampling_layers-i-1)),
                          order=i+configs.timekan.begin_order+1)
                    for i in range(configs.timekan.down_sampling_layers)
                ])
     
    def forward(self, level_list):
        level_list_reverse = level_list.copy()
        level_list_reverse.reverse()
        out_low = level_list_reverse[0]
        out_high = level_list_reverse[1]
        out_low = self.front_block(out_low)
        out_level_list = [out_low]
        for i in range(len(level_list_reverse) - 1):
            out_high = self.front_blocks[i](out_high)
            out_high_res = self.frequency_interpolation(out_low.transpose(1,2),
                                            self.configs.timekan.seq_len // (self.configs.timekan.down_sampling_window ** (self.configs.timekan.down_sampling_layers-i)),
                                            self.configs.timekan.seq_len // (self.configs.timekan.down_sampling_window ** (self.configs.timekan.down_sampling_layers-i-1))
                                            ).transpose(1,2)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(level_list_reverse) - 1:
                out_high = level_list_reverse[i + 2]    
            out_level_list.append(out_low) 
        out_level_list.reverse()
        return out_level_list

    def frequency_interpolation(self,x,seq_len,target_len):
        len_ratio = seq_len/target_len
        x_fft = torch.fft.rfft(x, dim=2)
        out_fft = torch.zeros([x_fft.size(0),x_fft.size(1),target_len//2+1],dtype=x_fft.dtype).to(x_fft.device)
        out_fft[:,:,:seq_len//2+1] = x_fft
        out = torch.fft.irfft(out_fft, dim=2)
        out = out * len_ratio
        return out
    
class M_KAN(nn.Module):
    def __init__(self,d_model,seq_len,order):
        super().__init__()
        self.channel_mixer = nn.Sequential(
            ChebyKANLayer(d_model, d_model,order)
        )
        self.conv = BasicConv(d_model,d_model,kernel_size=3,degree=order,groups=d_model)
    def forward(self,x):
        x1 = self.channel_mixer(x)
        x2 = self.conv(x)
        out  = x1 + x2
        return out 

class BasicConv(nn.Module):
    def __init__(self,c_in,c_out, kernel_size, degree,stride=1, padding=0, dilation=1, groups=1, act=False, bn=False, bias=False,dropout=0.):
        super(BasicConv, self).__init__()
        self.out_channels = c_out
        self.conv = nn.Conv1d(c_in,c_out, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(c_out) if bn else None
        self.act = nn.GELU() if act else None
        self.dropout = nn.Dropout(dropout)
    def forward(self, x): 
        if self.bn is not None:
            x = self.bn(x)
        x = self.conv(x.transpose(-1,-2)).transpose(-1,-2)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


# for PatchMLP insertions
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

    def __init__(self, d_model, enc_in):
        super().__init__()
        self.norm1 = nn.LayerNorm(enc_in)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff1 = nn.Sequential(
            nn.Linear(enc_in, enc_in),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.ff2 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ff1[0].weight, gain=1e-8)  # weight scale (gain)
        nn.init.constant_(self.ff1[0].bias, 0)
        nn.init.xavier_uniform_(self.ff2[0].weight, gain=1e-8)  # weight scale (gain)
        nn.init.constant_(self.ff2[0].bias, 0)

    def forward(self, x):
        y_0 = self.ff1(x)
        y_0 = y_0 + x
        y_0 = self.norm1(y_0)
        y_1 = y_0.permute(0, 2, 1)
        #y_1 = y_0.permute(0, 1, 2)
        y_1 = self.ff2(y_1)
        y_1 = y_1.permute(0, 2, 1)
        #y_1 = y_1.permute(0, 1, 2)
        y_2 = y_1 * y_0 + x
        y_2 = self.norm1(y_2)

        return y_2

class Encoder_Old(nn.Module):

    def __init__(self, d_model, enc_in):
        super().__init__()
        self.norm1 = nn.LayerNorm(enc_in)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff1 = nn.Sequential(
            nn.Linear(enc_in, enc_in),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.ff2 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ff1[0].weight, gain=1e-8)  # weight scale (gain)
        nn.init.constant_(self.ff1[0].bias, 0)
        nn.init.xavier_uniform_(self.ff2[0].weight, gain=1e-8)  # weight scale (gain)
        nn.init.constant_(self.ff2[0].bias, 0)

    def forward(self, x):
        y_0 = self.ff1(x)
        y_0 = y_0 + x
        y_0 = self.norm1(y_0)
        y_1 = y_0.permute(0, 1, 2)
        y_1 = self.ff2(y_1)
        y_1 = y_1.permute(0, 1, 2)
        y_2 = y_1 * y_0 + x
        y_2 = self.norm1(y_2)

        return y_2
# -------------------------

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.timekan.task_name
        self.seq_len = configs.timekan.seq_len
        self.label_len = configs.timekan.label_len
        self.pred_len = configs.timekan.pred_len
        self.down_sampling_window = configs.timekan.down_sampling_window
        self.channel_independence = configs.timekan.channel_independence
        self.res_blocks = nn.ModuleList([FrequencyDecomp(configs)
                                         for _ in range(configs.timekan.e_layers)])
        self.add_blocks = nn.ModuleList([FrequencyMixing(configs)
                                         for _ in range(configs.timekan.e_layers)])

        self.preprocess = series_decomp(configs.timekan.moving_avg)
        self.enc_in = configs.timekan.enc_in
        self.use_future_temporal_feature = configs.timekan.use_future_temporal_feature


        self.enc_embedding = DataEmbedding_wo_pos(1, configs.timekan.d_model, configs.timekan.embed, configs.timekan.freq,
                                                      configs.timekan.dropout)
        self.layer = configs.timekan.e_layers
        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.timekan.enc_in, affine=True, non_norm=True if configs.timekan.use_norm == 0 else False)
                for i in range(configs.timekan.down_sampling_layers + 1)
            ]
        )
        self.projection_layer = nn.Linear(
                    configs.timekan.d_model, 1, bias=True)
        self.predict_layer =nn. Linear(
                        configs.timekan.seq_len,
                        configs.timekan.pred_len,
                    )

        # PatchMLP Decomp & MLP
        self.decomposition = series_decomp(13)
        self.seasonal_layers = nn.ModuleList([
            Encoder_Old(configs.timekan.d_model, configs.timekan.enc_in)
            for i in range(configs.timekanmlp.e_layers)
        ])
        self.trend_layers = nn.ModuleList([
            Encoder_Old(configs.timekan.d_model, configs.timekan.enc_in)
            for i in range(configs.timekanmlp.e_layers)
        ])

        # TemporalSpatialDecomp
        self.TSDecomp = TemporalSpatialDecomp(configs)

        self.temporal_layers = nn.ModuleList([
            Encoder(configs.kanfusemlp.temporal_dim, configs.kanfusemlp.h_dim)
            for i in range(configs.kanfusemlp.mlp_blocks)
        ])
        self.spatial_layers = nn.ModuleList([
            Encoder(configs.kanfusemlp.spatial_dim, configs.kanfusemlp.h_dim)
            for i in range(configs.kanfusemlp.mlp_blocks)
        ])

        # learnable fusion parameters
        self.alpha = nn.Parameter(torch.rand(configs.kanfusemlp.temporal_dim, configs.kanfusemlp.h_dim))
        self.beta = nn.Parameter(torch.rand(configs.kanfusemlp.spatial_dim, configs.kanfusemlp.h_dim))

        # learnable moving avg parameters
        #self.gamma = nn.Parameter(torch.rand(configs.kanfusemlp.temporal_dim, configs.kanfusemlp.spatial_dim))
        #self.theta = nn.Parameter(torch.rand(configs.kanfusemlp.temporal_dim, configs.kanfusemlp.spatial_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.projection_layer.weight, gain=1e-7)  # weight scale (gain)
        nn.init.constant_(self.projection_layer.bias, 0)
        nn.init.xavier_uniform_(self.predict_layer.weight, gain=1e-7)  # weight scale (gain)
        nn.init.constant_(self.predict_layer.bias, 0)

    def forecast(self, x_enc):
        x_enc = self.__multi_level_process_inputs(x_enc)
        x_list = []
        for i, x in zip(range(len(x_enc)), x_enc, ):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

       
        enc_out_list = []
        for i, x in zip(range(len(x_list)), x_list):
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)


        for i in range(self.layer):
            enc_out_list = self.res_blocks[i](enc_out_list)
            enc_out_list = self.add_blocks[i](enc_out_list)

        dec_out = enc_out_list[0]
        dec_out = self.predict_layer(dec_out.permute(0, 2, 1)).permute(
                0, 2, 1)


        # PatchMLP Decomp & MLP
        seasonal_init, trend_init = self.decomposition(dec_out)
        #dec_out = seasonal_init + trend_init # * self.gamma + trend_init * self.theta
        #temporal_proj, spatial_proj = self.TSDecomp(dec_out)

        for mod in self.seasonal_layers:
            seasonal_init = mod(seasonal_init)
        for mod in self.trend_layers:
            trend_init = mod(trend_init)

        dec_out = seasonal_init + trend_init
        temporal_proj, spatial_proj = self.TSDecomp(dec_out)

        for mod in self.spatial_layers:
            spatial_proj = mod(spatial_proj)
        for mod in self.temporal_layers:
            temporal_proj = mod(temporal_proj)

        #temporal_proj = self.alpha(temporal_proj)
        #spatial_proj = self.beta(spatial_proj)
        temporal_proj = temporal_proj * self.alpha
        spatial_proj = spatial_proj * self.beta

        # project back into original shape
        dec_out = torch.bmm(temporal_proj, spatial_proj.transpose(1,2))

        #dec_out = seasonal_init * self.alpha + trend_init * self.beta
        # -----------------------

        dec_out = self.projection_layer(dec_out).reshape(B, self.configs.timekan.c_out, self.pred_len).permute(0, 2, 1).contiguous()
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out
    

    def __multi_level_process_inputs(self, x_enc):
        down_pool = torch.nn.AvgPool1d(self.configs.timekan.down_sampling_window)
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc
        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        for i in range(self.configs.timekan.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling
        x_enc = x_enc_sampling_list
        return x_enc

    def forward(self, x_enc):#, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out
        else:
            raise ValueError('Other tasks implemented yet')


