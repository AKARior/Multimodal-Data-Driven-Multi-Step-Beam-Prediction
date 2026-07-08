"""Local model building blocks used by ``model.py``.

This file vendors the small subset of reusable modules that were previously
imported from project-root experiment scripts. Keeping them here makes the
paper model self-contained and easier to inspect inside one folder.
"""

import math

import torch
import torch.nn as nn

from mamba_ssm import Mamba2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        mean = (x**2).mean(dim=-1, keepdim=True)
        x_norm = x / torch.sqrt(mean + self.eps)
        return self.gamma * x_norm


def sinkhorn_knopp_batched(A, it=20, eps=1e-8):
    """Approximate doubly-stochastic normalization for batched matrices."""
    batch_size, n, _ = A.shape
    u = torch.ones(batch_size, n, device=A.device, dtype=A.dtype)
    v = torch.ones(batch_size, n, device=A.device, dtype=A.dtype)

    for _ in range(it):
        Av = torch.bmm(A, v.unsqueeze(2)).squeeze(2)
        u = 1.0 / (Av + eps)

        At_u = torch.bmm(A.transpose(1, 2), u.unsqueeze(2)).squeeze(2)
        v = 1.0 / (At_u + eps)

    U = torch.diag_embed(u)
    V = torch.diag_embed(v)
    P = torch.bmm(torch.bmm(U, A), V)
    return P, U, V


class ManifoldHyperConnectionFuse(nn.Module):
    """Manifold hyper-connection block used by the vision residual Mamba path."""

    def __init__(self, dim, rate=2, max_sk_it=20):
        super().__init__()
        self.n = rate
        self.dim = dim
        self.nc = self.n * self.dim
        self.n2 = self.n * self.n
        self.max_sk_it = max_sk_it

        self.norm = RMSNorm(self.nc)
        self.w = nn.Parameter(torch.zeros(self.nc, self.n2 + 2 * self.n))
        self.alpha = nn.Parameter(torch.ones(3) * 0.01)
        self.beta = nn.Parameter(torch.zeros(self.n2 + 2 * self.n) * 0.01)

    def mapping(self, h):
        batch_size, seq_len, n_paths, dim = h.shape
        h_vec_flat = h.reshape(batch_size, seq_len, n_paths * dim)
        h_vec = self.norm.gamma * h_vec_flat
        H = h_vec @ self.w

        radius = h_vec_flat.norm(dim=-1, keepdim=True) / math.sqrt(self.nc)
        radius_inv = 1.0 / (radius + 1e-8)

        H_pre = radius_inv * H[:, :, :n_paths] * self.alpha[0] + self.beta[:n_paths]
        H_post = radius_inv * H[:, :, n_paths : 2 * n_paths] * self.alpha[1] + self.beta[n_paths : 2 * n_paths]
        H_res = radius_inv * H[:, :, 2 * n_paths :] * self.alpha[2] + self.beta[2 * n_paths :]

        H_pre = torch.sigmoid(H_pre)
        H_post = 2.0 * torch.sigmoid(H_post)

        H_res = H_res.reshape(batch_size, seq_len, n_paths, n_paths)
        H_res_exp = H_res.exp()
        with torch.no_grad():
            _, U, V = sinkhorn_knopp_batched(
                H_res_exp.reshape(batch_size * seq_len, n_paths, n_paths),
                it=self.max_sk_it,
            )
        P = torch.bmm(
            torch.bmm(U.detach(), H_res_exp.reshape(batch_size * seq_len, n_paths, n_paths)),
            V.detach(),
        )
        H_res = P.reshape(batch_size, seq_len, n_paths, n_paths)

        return H_pre, H_post, H_res

    @staticmethod
    def process(h, H_pre, H_res):
        h_pre = H_pre.unsqueeze(2) @ h
        h_res = H_res @ h
        return h_pre, h_res

    @staticmethod
    def depth_connection(h_res, h_out, beta):
        post_mapping = beta.unsqueeze(-1) @ h_out
        return post_mapping + h_res


def build_residual_mamba_layers(
    num_layers,
    d_model,
    residual_mode="standard",
    use_sub_layer_norm=True,
    mhc_rate=2,
    mhc_max_sk_it=20,
):
    """Build reusable Mamba residual layers with standard or mHC residuals."""
    if residual_mode not in ("standard", "mhc"):
        raise ValueError("residual_mode must be one of ['standard', 'mhc']")

    norm_module = (lambda: nn.LayerNorm(d_model)) if use_sub_layer_norm else (lambda: nn.Identity())
    layers = []
    for _ in range(num_layers):
        if residual_mode == "standard":
            layers.append(
                nn.ModuleDict(
                    {
                        "norm": norm_module(),
                        "mamba": Mamba2(
                            d_model=d_model,
                            d_state=64,
                            d_conv=4,
                            expand=2,
                            headdim=2,
                        ),
                    }
                )
            )
        else:
            layers.append(
                nn.ModuleDict(
                    {
                        "mhc": ManifoldHyperConnectionFuse(dim=d_model, rate=mhc_rate, max_sk_it=mhc_max_sk_it),
                        "norm": norm_module(),
                        "mamba": Mamba2(
                            d_model=d_model,
                            d_state=64,
                            d_conv=4,
                            expand=2,
                            headdim=2,
                        ),
                    }
                )
            )
    return nn.ModuleList(layers)


def forward_residual_mamba(x, enc_layers, dropout, residual_mode="standard"):
    """Run the reusable residual-Mamba encoder stack."""
    if len(enc_layers) == 0:
        return x

    if residual_mode == "standard":
        for layer in enc_layers:
            x = x + dropout(layer["mamba"](layer["norm"](x)))
        return x

    if residual_mode != "mhc":
        raise ValueError("residual_mode must be one of ['standard', 'mhc']")

    rate = int(enc_layers[0]["mhc"].n)
    h = x.unsqueeze(2).repeat(1, 1, rate, 1)
    for layer in enc_layers:
        H_pre, H_post, H_res = layer["mhc"].mapping(h)
        h_pre, h_res = layer["mhc"].process(h, H_pre, H_res)
        sub_in = layer["norm"](h_pre.squeeze(2))
        h_out = dropout(layer["mamba"](sub_in)).unsqueeze(2)
        h = layer["mhc"].depth_connection(h_res, h_out, beta=H_post)
    return h.sum(dim=2)


class TemporalRefinerBlock(nn.Module):
    def __init__(self, d_model, hidden_dim=20, kernel_size=3, dropout=0.1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
            bias=False,
        )
        self.channel_norm = nn.LayerNorm(d_model)
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        temporal = self.temporal_norm(x).transpose(1, 2)
        temporal = self.temporal_conv(temporal).transpose(1, 2)
        x = x + self.dropout(temporal)
        x = x + self.dropout(self.channel_mlp(self.channel_norm(x)))
        return x


class OverlapPatchMixerBranch(nn.Module):
    def __init__(self, input_dim, seq_len, win_len, stride, hidden_dim):
        super().__init__()
        if win_len > seq_len:
            raise ValueError("win_len must be <= seq_len")
        if stride <= 0:
            raise ValueError("stride must be > 0")

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.win_len = win_len
        self.stride = stride
        self.num_patches = max(1, math.ceil((seq_len - win_len) / stride) + 1)
        self.cover_len = (self.num_patches - 1) * stride + win_len
        self.pad_len = max(0, self.cover_len - seq_len)
        self.patch_dim = input_dim * win_len

        self.patch_encode = nn.Linear(self.patch_dim, hidden_dim, bias=False)
        self.patch_mix = nn.Linear(self.num_patches, self.num_patches, bias=False)
        self.patch_decode = nn.Linear(hidden_dim, self.patch_dim, bias=False)
        self.out_norm = nn.LayerNorm(input_dim)

        nn.init.xavier_uniform_(self.patch_encode.weight)
        nn.init.xavier_uniform_(self.patch_mix.weight)
        nn.init.xavier_uniform_(self.patch_decode.weight)

    def _pad_sequence(self, x):
        if self.pad_len == 0:
            return x
        pad = x[:, -1:, :].expand(-1, self.pad_len, -1)
        return torch.cat([x, pad], dim=1)

    def _extract_patches(self, x):
        x_pad = self._pad_sequence(x)
        patches = x_pad.transpose(1, 2).unfold(dimension=-1, size=self.win_len, step=self.stride)
        patches = patches.permute(0, 2, 1, 3).contiguous()
        return patches.reshape(x.size(0), self.num_patches, self.patch_dim)

    def _overlap_add(self, patch_values, batch_size, device, dtype):
        patch_values = patch_values.view(batch_size, self.num_patches, self.input_dim, self.win_len)
        patch_values = patch_values.permute(0, 1, 3, 2).contiguous()

        recon = torch.zeros(batch_size, self.cover_len, self.input_dim, device=device, dtype=dtype)
        counts = torch.zeros(batch_size, self.cover_len, 1, device=device, dtype=dtype)

        for patch_idx in range(self.num_patches):
            start = patch_idx * self.stride
            end = start + self.win_len
            recon[:, start:end, :] += patch_values[:, patch_idx, :, :]
            counts[:, start:end, :] += 1

        recon = recon / counts.clamp_min(1.0)
        return recon[:, : self.seq_len, :]

    def forward(self, x):
        patches = self._extract_patches(x)
        local_tokens = self.patch_encode(patches)
        mixed_tokens = self.patch_mix(local_tokens.transpose(1, 2)).transpose(1, 2)
        decoded_patches = self.patch_decode(local_tokens + mixed_tokens)
        time_out = self._overlap_add(decoded_patches, x.size(0), x.device, x.dtype)
        return self.out_norm(time_out)


class OverlapMixLinearNumericEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        seq_len,
        d_model,
        win_len=5,
        stride=1,
        time_hidden=32,
        aux_win_len=3,
        aux_time_hidden=16,
        lpf=3,
        freq_rank=2,
        dropout=0.1,
        refiner_hidden=20,
        temporal_kernel=3,
        use_freq_branch=True,
        use_aux_branch=True,
    ):
        super().__init__()
        if win_len > seq_len:
            raise ValueError("win_len must be <= seq_len")
        if stride <= 0:
            raise ValueError("stride must be > 0")

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.win_len = win_len
        self.stride = stride
        self.num_patches = max(1, math.ceil((seq_len - win_len) / stride) + 1)
        self.cover_len = (self.num_patches - 1) * stride + win_len
        self.pad_len = max(0, self.cover_len - seq_len)
        self.aux_win_len = min(seq_len, aux_win_len)
        self.use_freq_branch = use_freq_branch
        self.use_aux_branch = use_aux_branch

        self.patch_dim = self.input_dim * win_len
        self.time_hidden = time_hidden
        self.freq_bins = seq_len // 2 + 1
        self.lpf = min(lpf, self.freq_bins)
        self.freq_rank = min(freq_rank, self.lpf)

        self.patch_encode = nn.Linear(self.patch_dim, time_hidden, bias=False)
        self.patch_mix = nn.Linear(self.num_patches, self.num_patches, bias=False)
        self.patch_decode = nn.Linear(time_hidden, self.patch_dim, bias=False)
        self.time_norm = nn.LayerNorm(self.input_dim)
        self.aux_time_branch = (
            OverlapPatchMixerBranch(
                input_dim=self.input_dim,
                seq_len=seq_len,
                win_len=self.aux_win_len,
                stride=stride,
                hidden_dim=aux_time_hidden,
            )
            if self.use_aux_branch and self.aux_win_len != win_len
            else None
        )
        self.time_fuse_gate = nn.Linear(2 * self.input_dim, self.input_dim) if self.aux_time_branch is not None else None

        if self.use_freq_branch:
            self.freq_down_real = nn.Linear(self.lpf, self.freq_rank, bias=False)
            self.freq_up_real = nn.Linear(self.freq_rank, self.freq_bins, bias=False)
            self.freq_down_imag = nn.Linear(self.lpf, self.freq_rank, bias=False)
            self.freq_up_imag = nn.Linear(self.freq_rank, self.freq_bins, bias=False)
            self.freq_norm = nn.LayerNorm(self.input_dim)
            self.freq_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.freq_down_real = None
            self.freq_up_real = None
            self.freq_down_imag = None
            self.freq_up_imag = None
            self.freq_norm = None
            self.freq_scale = None

        self.time_scale = nn.Parameter(torch.tensor(0.5))

        self.input_norm = nn.LayerNorm(self.input_dim)
        self.out_proj = nn.Linear(self.input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.refiner = TemporalRefinerBlock(
            d_model=d_model,
            hidden_dim=refiner_hidden,
            kernel_size=temporal_kernel,
            dropout=dropout,
        )
        self.out_dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.patch_encode.weight)
        nn.init.xavier_uniform_(self.patch_mix.weight)
        nn.init.xavier_uniform_(self.patch_decode.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.time_fuse_gate is not None:
            nn.init.xavier_uniform_(self.time_fuse_gate.weight)

    def _pad_sequence(self, x):
        if self.pad_len == 0:
            return x
        pad = x[:, -1:, :].expand(-1, self.pad_len, -1)
        return torch.cat([x, pad], dim=1)

    def _extract_patches(self, x):
        x_pad = self._pad_sequence(x)
        patches = x_pad.transpose(1, 2).unfold(dimension=-1, size=self.win_len, step=self.stride)
        patches = patches.permute(0, 2, 1, 3).contiguous()
        return patches.reshape(x.size(0), self.num_patches, self.patch_dim)

    def _overlap_add(self, patch_values, batch_size, device, dtype):
        patch_values = patch_values.view(batch_size, self.num_patches, self.input_dim, self.win_len)
        patch_values = patch_values.permute(0, 1, 3, 2).contiguous()

        recon = torch.zeros(batch_size, self.cover_len, self.input_dim, device=device, dtype=dtype)
        counts = torch.zeros(batch_size, self.cover_len, 1, device=device, dtype=dtype)

        for patch_idx in range(self.num_patches):
            start = patch_idx * self.stride
            end = start + self.win_len
            recon[:, start:end, :] += patch_values[:, patch_idx, :, :]
            counts[:, start:end, :] += 1

        recon = recon / counts.clamp_min(1.0)
        return recon[:, : self.seq_len, :]

    def _time_branch(self, x):
        patches = self._extract_patches(x)
        local_tokens = self.patch_encode(patches)
        mixed_tokens = self.patch_mix(local_tokens.transpose(1, 2)).transpose(1, 2)
        decoded_patches = self.patch_decode(local_tokens + mixed_tokens)
        time_out = self.time_norm(self._overlap_add(decoded_patches, x.size(0), x.device, x.dtype))
        if self.aux_time_branch is None:
            return time_out

        aux_time_out = self.aux_time_branch(x)
        gate = torch.sigmoid(self.time_fuse_gate(torch.cat([time_out, aux_time_out], dim=-1)))
        fused_time = gate * time_out + (1.0 - gate) * aux_time_out
        return self.time_norm(fused_time)

    def _freq_branch(self, x):
        if not self.use_freq_branch:
            raise RuntimeError("Frequency branch is disabled.")
        centered = x - x.mean(dim=1, keepdim=True)
        x_fft = torch.fft.rfft(centered, dim=1)
        low_fft = x_fft[:, : self.lpf, :]

        low_real = low_fft.real.permute(0, 2, 1)
        low_imag = low_fft.imag.permute(0, 2, 1)

        full_real = self.freq_up_real(self.freq_down_real(low_real))
        full_imag = self.freq_up_imag(self.freq_down_imag(low_imag))
        filtered = torch.complex(full_real, full_imag).permute(0, 2, 1)

        freq_out = torch.fft.irfft(filtered, n=self.seq_len, dim=1)
        return self.freq_norm(freq_out)

    def forward(self, numeric):
        residual = self.input_norm(numeric)
        time_out = self._time_branch(residual)
        fused = residual + torch.tanh(self.time_scale) * time_out
        if self.use_freq_branch:
            freq_out = self._freq_branch(residual)
            fused = fused + torch.tanh(self.freq_scale) * freq_out
        fused = self.out_dropout(fused)
        fused = self.out_proj(fused)
        fused = self.pos_encoder(fused)
        return self.refiner(fused)
