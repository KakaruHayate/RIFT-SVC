# refer to： 
# https://github.com/CNChTu/Diffusion-SVC/blob/v2.0_dev/diffusion/naive_v2/model_conformer_naive.py
# https://github.com/CNChTu/Diffusion-SVC/blob/v2.0_dev/diffusion/naive_v2/naive_v2_diff.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat
from rift_svc.modules import SinusPositionEmbedding


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)


class SwiGLU(nn.Module):
    # Swish-Applies the gated linear unit function.
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # out, gate = x.chunk(2, dim=self.dim)
        # Using torch.split instead of chunk for ONNX export compatibility.
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        return out * F.silu(gate)


class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


class LYNXConvModule(nn.Module):
    @staticmethod
    def calc_same_padding(kernel_size):
        pad = kernel_size // 2
        return pad, pad - (kernel_size + 1) % 2

    def __init__(self, dim, expansion_factor, kernel_size=31, activation='PReLU', dropout=0.):
        super().__init__()
        inner_dim = dim * expansion_factor
        activation_classes = {
            'SiLU': nn.SiLU,
            'ReLU': nn.ReLU,
            'PReLU': lambda: nn.PReLU(inner_dim)
        }
        activation = activation if activation is not None else 'PReLU'
        if activation not in activation_classes:
            raise ValueError(f'{activation} is not a valid activation')
        _activation = activation_classes[activation]()
        padding = self.calc_same_padding(kernel_size)
        if float(dropout) > 0.:
            _dropout = nn.Dropout(dropout)
        else:
            _dropout = nn.Identity()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose((1, 2)),
            nn.Conv1d(dim, inner_dim * 2, 1),
            SwiGLU(dim=1),
            nn.Conv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding[0], groups=inner_dim),
            _activation,
            nn.Conv1d(inner_dim, dim, 1),
            Transpose((1, 2)),
            _dropout
        )

    def forward(self, x):
        return self.net(x)


class LYNXNetResidualLayer(nn.Module):
    def __init__(self, dim_cond, dim, expansion_factor, kernel_size=31, activation='PReLU', dropout=0.):
        super().__init__()
        self.diffusion_projection = nn.Conv1d(dim, dim, 1)
        self.conditioner_projection = nn.Conv1d(dim_cond, dim, 1)
        self.convmodule = LYNXConvModule(dim=dim, expansion_factor=expansion_factor, kernel_size=kernel_size,
                                         activation=activation, dropout=dropout)

    def forward(self, x, conditioner, diffusion_step):
        x = x + self.conditioner_projection(conditioner)
        res_x = x
        x = x + self.diffusion_projection(diffusion_step)
        x = x.transpose(1, 2)
        x = self.convmodule(x)  # (#batch, dim, length) 
        x = x.transpose(1, 2) + res_x

        return x  # (#batch, length, dim)


class LYNXNet(nn.Module):
    def __init__(self, num_channels=512, num_layers=6, mel_channels=128, expansion_factor=2, kernel_size=31,
                 activation='PReLU', dropout=0.):
        """
        LYNXNet(Linear Gated Depthwise Separable Convolution Network)
        TIPS:You can control the style of the generated results by modifying the 'activation', 
            - 'PReLU'(default) : Similar to WaveNet
            - 'SiLU' : Voice will be more pronounced, not recommended for use under DDPM
            - 'ReLU' : Contrary to 'SiLU', Voice will be weakened
        """
        super().__init__()
        self.input_projection = Conv1d(mel_channels, num_channels, 1)
        self.diffusion_embedding = nn.Sequential(
            SinusPositionEmbedding(num_channels),
            nn.Linear(num_channels, num_channels * 4),
            nn.GELU(),
            nn.Linear(num_channels * 4, num_channels),
        )
        self.residual_layers = nn.ModuleList(
            [
                LYNXNetResidualLayer(
                    dim_cond=num_channels,
                    dim=num_channels,
                    expansion_factor=expansion_factor,
                    kernel_size=kernel_size,
                    activation=activation,
                    dropout=dropout
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(num_channels)
        self.output_projection = Conv1d(num_channels, mel_channels, kernel_size=1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, x, diffusion_step, cond):
        """
        :param x: [B, M, T]
        :param diffusion_step: [B, ]
        :param cond: [B, H, T]
        :return:
        """

        x = self.input_projection(x)  # x [B, residual_channel, T]
        # x = F.gelu(x) # 去掉这个ACT效果更好些

        diffusion_step = self.diffusion_embedding(diffusion_step).unsqueeze(-1)

        for layer in self.residual_layers:
            x = layer(x, cond, diffusion_step)

        # post-norm
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)

        # MLP and GLU
        x = self.output_projection(x)  # [B, M, T]

        return x


class LYNXNet4RIFT(nn.Module):
    def __init__(self, num_channels=1024, num_layers=6, expansion_factor=2, kernel_size=31, num_speaker=1, mel_channels=128, cvec_dim=768, whisper_dim=1280):
        """
        Initialize the LYNXNet4RIFT module.
        
        :param num_channels: Number of channels in the network.
        :param num_layers: Number of layers in the LYNXNet.
        :param expansion_factor: Expansion factor for the LYNXNet.
        :param kernel_size: Kernel size for the LYNXNet.
        :param num_speaker: Number of speakers for the speaker embedding.
        :param mel_channels: Number of mel channels in the input.
        :param cvec_dim: Dimension of the contentvec.
        :param whisper_dim: Dimension of the whisper.
        """
        super().__init__()
        self.f0_embed = nn.Linear(1, num_channels)
        self.rms_embed = nn.Linear(1, num_channels)
        self.cvec_embed = nn.Linear(cvec_dim, num_channels)
        self.whisper_embed = nn.Linear(whisper_dim, num_channels)
        self.gate_linear = nn.Linear(2*num_channels, num_channels)
        self.lynxnet = LYNXNet(num_channels, num_layers, mel_channels, expansion_factor, kernel_size)
        self.spk_embed = nn.Embedding(num_speaker, num_channels)
        self.null_whisper_embed = nn.Embedding(1, whisper_dim)

    def forward(self, x, spk, f0, rms, cvec, whisper, time, drop_whisper, mask):
        """
        Forward pass of the LYNXNet4RIFT module.
        
        :param x: Input tensor of shape (batch, seq_len, num_channels).
        :param spk: Speaker indices tensor of shape (batch,).
        :param f0: F0 tensor of shape (batch, seq_len).
        :param rms: RMS tensor of shape (batch, seq_len).
        :param cvec: contentvec tensor of shape (batch, seq_len, cvec_dim).
        :param whisper: Whisper tensor of shape (batch, seq_len, whisper_dim).
        :param time: Time step tensor of shape (batch,).
        :param drop_whisper: Boolean or tensor indicating whether to drop the whisper embedding.
        :param mask: Mask tensor of shape (batch, seq_len) to apply to the output.
        
        :return: Output tensor after processing through the LYNXNet, multiplied by the mask.
        """
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = repeat(time, ' -> b', b = batch)
        if f0.ndim == 2:
            f0 = f0.unsqueeze(-1)
        if rms.ndim == 2:
            rms = rms.unsqueeze(-1)
        f0_embed = self.f0_embed(f0 / 1200)
        rms_embed = self.rms_embed(rms)
        cvec_embed = self.cvec_embed(cvec)
        if isinstance(drop_whisper, bool):
            drop_whisper = torch.full((batch,), drop_whisper, device=x.device)
        null_whisper = repeat(self.null_whisper_embed.weight, '1 d -> b n d', b=batch, n=seq_len)
        whisper = torch.where(
            drop_whisper.unsqueeze(-1).unsqueeze(-1),  # [b, 1, 1]
            null_whisper,                              # [b, n, d]
            whisper                            # [b, n, d]
        )
        whisper_embed = self.whisper_embed(whisper)
        gate = F.sigmoid(self.gate_linear(torch.cat((cvec_embed, whisper_embed), dim=-1)))
        cond = f0_embed + rms_embed + gate * cvec_embed + (1 - gate) * whisper_embed
        
        # 以上参考原来的实现
        spk_embeds = self.spk_embed(spk).unsqueeze(1)
        cond = cond + spk_embeds
        # 这里的DiT上spk emb嵌入到了t上，这里参考以往的做法嵌入到condition

        output = self.lynxnet(x=x.transpose(1, 2), diffusion_step=time, cond=cond.transpose(1, 2)).transpose(1, 2)
        # 维度全是反的所以直接调转处理

        
        return output * mask.unsqueeze(-1) # 处理下输出的mask
