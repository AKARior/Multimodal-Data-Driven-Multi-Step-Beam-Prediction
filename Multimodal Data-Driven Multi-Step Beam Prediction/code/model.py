import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

try:
    from .model_components import (
        OverlapMixLinearNumericEncoder,
        PositionalEncoding,
        build_residual_mamba_layers,
        forward_residual_mamba,
    )
except ImportError:
    from model_components import (
        OverlapMixLinearNumericEncoder,
        PositionalEncoding,
        build_residual_mamba_layers,
        forward_residual_mamba,
    )


class GeMPool2d(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p_init)
        self.eps = eps

    def forward(self, x):
        p = torch.clamp(self.p, min=1.0, max=8.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=(-1, -2), keepdim=True)
        return x.pow(1.0 / p)


class VisionEncoder(nn.Module):
    """Image encoder: ResNet18 + pooling + Linear + LayerNorm + residual Mamba + output LN."""

    def __init__(
        self,
        d_model=128,
        num_layers=2,
        pooling_type="gem",
        residual_mode="mhc",
        use_sub_layer_norm=False,
        mhc_rate=2,
        mhc_max_sk_it=20,
        dropout=0.1,
    ):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if pooling_type == "gem":
            base.avgpool = GeMPool2d(p_init=3.0)
        elif pooling_type == "avg":
            base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError("pooling_type must be one of ['gem', 'avg']")
        base.fc = nn.Identity()
        self.resnet = base
        self.pooling_type = pooling_type
        self.resnet_fc = nn.Linear(512, d_model)
        nn.init.xavier_uniform_(self.resnet_fc.weight)
        nn.init.zeros_(self.resnet_fc.bias)

        self.feat_norm = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.residual_mode = residual_mode
        self.enc_layers = build_residual_mamba_layers(
            num_layers=num_layers,
            d_model=d_model,
            residual_mode=residual_mode,
            use_sub_layer_norm=use_sub_layer_norm,
            mhc_rate=mhc_rate,
            mhc_max_sk_it=mhc_max_sk_it,
        )
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, image):
        bsz, t_img, c, h, w = image.shape
        image_input = image.view(bsz * t_img, c, h, w)
        feature = self.resnet(image_input).view(bsz, t_img, -1)
        feature = self.resnet_fc(feature)
        feature = self.feat_norm(feature)
        x = self.pos_encoder(feature)
        x = forward_residual_mamba(x, self.enc_layers, self.dropout, residual_mode=self.residual_mode)
        return self.out_norm(x)


class NumericEncoder(nn.Module):
    """Numeric encoder: overlap temporal mixing + low-pass FFT branch + output LN."""

    def __init__(
        self,
        input_dim=5,
        seq_len=10,
        d_model=128,
        win_len=5,
        stride=1,
        time_hidden=48,
        aux_win_len=3,
        aux_time_hidden=24,
        lpf=3,
        freq_rank=3,
        dropout=0.03,
        refiner_hidden=20,
        temporal_kernel=3,
        use_freq_branch=True,
        use_aux_branch=True,
    ):
        super().__init__()
        self.encoder = OverlapMixLinearNumericEncoder(
            input_dim=input_dim,
            seq_len=seq_len,
            d_model=d_model,
            win_len=win_len,
            stride=stride,
            time_hidden=time_hidden,
            aux_win_len=aux_win_len,
            aux_time_hidden=aux_time_hidden,
            lpf=lpf,
            freq_rank=freq_rank,
            dropout=dropout,
            refiner_hidden=refiner_hidden,
            temporal_kernel=temporal_kernel,
            use_freq_branch=use_freq_branch,
            use_aux_branch=use_aux_branch,
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, numeric):
        return self.out_norm(self.encoder(numeric))


class DecoderModuleWithHidden(nn.Module):
    """Shared zero-query Transformer decoder."""

    def __init__(self, d_model, num_heads, num_layers, dropout, num_classes):
        super().__init__()
        self.pos_decoder = PositionalEncoding(d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model, num_heads, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.fc_out = nn.Linear(d_model, num_classes)
        nn.init.xavier_uniform_(self.fc_out.weight)
        if self.fc_out.bias is not None:
            nn.init.zeros_(self.fc_out.bias)

    def forward_hidden(self, memory, pred_len):
        batch_size = memory.size(0)
        tgt = torch.zeros(batch_size, pred_len, memory.size(-1), device=memory.device)
        tgt = self.pos_decoder(tgt)
        return self.decoder(tgt, memory)

    def forward(self, memory, pred_len):
        hidden = self.forward_hidden(memory, pred_len)
        return self.fc_out(hidden)


class DepthwiseTemporalConvBlock(nn.Module):
    def __init__(self, d_model, hidden_dim=None, kernel_size=3, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or (2 * d_model)
        padding = kernel_size // 2
        self.temporal_norm = nn.LayerNorm(d_model)
        self.dw_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding, groups=d_model)
        self.pw_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.channel_norm = nn.LayerNorm(d_model)
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x):
        y = self.temporal_norm(x).transpose(1, 2)
        y = self.pw_conv(F.gelu(self.dw_conv(y))).transpose(1, 2)
        x = x + y
        x = x + self.channel_mlp(self.channel_norm(x))
        return x


class StepConvLiteFusion(nn.Module):
    """Lightweight multimodal hidden fusion over the 5 prediction steps."""

    def __init__(
        self,
        d_model=128,
        pred_len=5,
        hidden_dim=None,
        kernel_size=3,
        dropout=0.1,
        use_diff_term=True,
        use_prod_term=True,
        anchor_mode="image",
        relation_only=False,
        normalize_relation_terms=False,
        diff_mode="signed",
    ):
        super().__init__()
        if anchor_mode not in {"image", "numeric", "symmetric", "adaptive"}:
            raise ValueError("anchor_mode must be one of ['image', 'numeric', 'symmetric', 'adaptive']")
        if diff_mode not in {"signed", "abs"}:
            raise ValueError("diff_mode must be one of ['signed', 'abs']")
        hidden_dim = hidden_dim or (2 * d_model)
        self.use_diff_term = use_diff_term
        self.use_prod_term = use_prod_term
        self.anchor_mode = anchor_mode
        self.relation_only = relation_only
        self.normalize_relation_terms = normalize_relation_terms
        self.diff_mode = diff_mode
        self.use_zero_relation_context = relation_only and (not use_diff_term) and (not use_prod_term)
        pair_dim = 0 if relation_only else 2 * d_model
        if use_diff_term:
            pair_dim += d_model
        if use_prod_term:
            pair_dim += d_model
        if pair_dim <= 0 and not self.use_zero_relation_context:
            raise ValueError("StepConvLiteFusion requires at least one pair input term")
        if normalize_relation_terms:
            self.img_rel_norm = nn.LayerNorm(d_model)
            self.num_rel_norm = nn.LayerNorm(d_model)
            self.diff_norm = nn.LayerNorm(d_model) if use_diff_term else nn.Identity()
            self.prod_norm = nn.LayerNorm(d_model) if use_prod_term else nn.Identity()
        else:
            self.img_rel_norm = nn.Identity()
            self.num_rel_norm = nn.Identity()
            self.diff_norm = nn.Identity()
            self.prod_norm = nn.Identity()
        if self.use_zero_relation_context:
            self.pair_proj = None
        else:
            self.pair_proj = nn.Sequential(
                nn.LayerNorm(pair_dim),
                nn.Linear(pair_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
            )
        self.step_tokens = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)
        self.step_block = DepthwiseTemporalConvBlock(
            d_model=d_model,
            hidden_dim=2 * d_model,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Sigmoid(),
        )
        self.delta = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        if anchor_mode == "adaptive":
            self.anchor_gate = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2 * d_model),
            )
        else:
            self.anchor_gate = None
        self.alpha = nn.Parameter(torch.tensor(0.14))
        self.out_norm = nn.LayerNorm(d_model)
        self.out_ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, image_hidden, numeric_hidden):
        image_rel = self.img_rel_norm(image_hidden)
        numeric_rel = self.num_rel_norm(numeric_hidden)
        pair_terms = [] if self.relation_only else [image_hidden, numeric_hidden]
        if self.use_diff_term:
            diff = image_rel - numeric_rel
            if self.diff_mode == "abs":
                diff = diff.abs()
            pair_terms.append(self.diff_norm(diff))
        if self.use_prod_term:
            pair_terms.append(self.prod_norm(image_rel * numeric_rel))
        if self.use_zero_relation_context:
            pair_ctx = torch.zeros_like(image_hidden)
        else:
            pair_input = torch.cat(pair_terms, dim=-1)
            pair_ctx = self.pair_proj(pair_input) + self.step_tokens[:, : image_hidden.size(1), :]
            pair_ctx = self.step_block(pair_ctx)
        fusion_input = torch.cat([image_hidden, numeric_hidden, pair_ctx], dim=-1)
        gate = self.gate(fusion_input)
        delta = self.delta(fusion_input)
        if self.anchor_mode == "image":
            anchor = image_hidden
        elif self.anchor_mode == "numeric":
            anchor = numeric_hidden
        elif self.anchor_mode == "adaptive":
            gate_logits = self.anchor_gate(pair_ctx).view(
                image_hidden.size(0),
                image_hidden.size(1),
                2,
                image_hidden.size(2),
            )
            gate_weights = torch.softmax(gate_logits, dim=2)
            anchor = gate_weights[:, :, 0, :] * image_hidden + gate_weights[:, :, 1, :] * numeric_hidden
        else:
            anchor = 0.5 * (image_hidden + numeric_hidden)
        fused = anchor + torch.tanh(self.alpha) * gate * delta
        fused = fused + self.out_ffn(self.out_norm(fused))
        return self.out_norm(fused)


class LiteStepMixHead(nn.Module):
    """Multi-only lightweight head after hidden fusion."""

    def __init__(self, d_model=128, num_classes=64, hidden_dim=None, kernel_size=3, dropout=0.1, pred_len=5):
        super().__init__()
        hidden_dim = hidden_dim or (2 * d_model)
        self.step_tokens = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)
        self.step_mixer = DepthwiseTemporalConvBlock(
            d_model=d_model,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.out_ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.classifier = nn.Linear(d_model, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def bootstrap_from_decoder(self, decoder_fc_out):
        with torch.no_grad():
            self.classifier.weight.copy_(decoder_fc_out.weight)
            if decoder_fc_out.bias is not None and self.classifier.bias is not None:
                self.classifier.bias.copy_(decoder_fc_out.bias)

    def forward(self, fused_hidden):
        x = fused_hidden + self.step_tokens[:, : fused_hidden.size(1), :]
        x = self.step_mixer(x)
        x = x + self.out_ffn(self.out_norm(x))
        x = self.out_norm(x)
        return self.classifier(x)


class LiteMultiBeamModel(nn.Module):
    """Paper version: lightweight fusion + shared Transformer multi-decoder."""

    def __init__(
        self,
        input_dim=5,
        seq_len=10,
        pred_len=5,
        d_model=128,
        num_heads=2,
        num_classes=64,
        image_enc_layers=2,
        dec_layers=2,
        decoder_dropout=0.03,
        cross_dropout=0.1,
        numeric_use_freq_branch=True,
        numeric_use_aux_branch=True,
        image_pooling_type="gem",
        image_residual_mode="mhc",
        image_use_sub_layer_norm=False,
        fusion_use_diff_term=True,
        fusion_use_prod_term=True,
        fusion_anchor_mode="image",
        fusion_relation_only=False,
        fusion_normalize_relation_terms=False,
        fusion_diff_mode="signed",
    ):
        super().__init__()
        self.pred_len = pred_len

        self.numeric = NumericEncoder(
            input_dim=input_dim,
            seq_len=seq_len,
            d_model=d_model,
            win_len=5,
            stride=1,
            time_hidden=48,
            aux_win_len=3,
            aux_time_hidden=24,
            lpf=3,
            freq_rank=3,
            dropout=0.03,
            refiner_hidden=20,
            temporal_kernel=3,
            use_freq_branch=numeric_use_freq_branch,
            use_aux_branch=numeric_use_aux_branch,
        )
        self.vision = VisionEncoder(
            d_model=d_model,
            num_layers=image_enc_layers,
            pooling_type=image_pooling_type,
            residual_mode=image_residual_mode,
            use_sub_layer_norm=image_use_sub_layer_norm,
            mhc_rate=2,
            mhc_max_sk_it=20,
            dropout=0.1,
        )
        self.decoder = DecoderModuleWithHidden(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=dec_layers,
            dropout=decoder_dropout,
            num_classes=num_classes,
        )
        self.multi_decoder = DecoderModuleWithHidden(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=dec_layers,
            dropout=decoder_dropout,
            num_classes=num_classes,
        )
        self.fusion = StepConvLiteFusion(
            d_model=d_model,
            pred_len=pred_len,
            hidden_dim=2 * d_model,
            kernel_size=3,
            dropout=cross_dropout,
            use_diff_term=fusion_use_diff_term,
            use_prod_term=fusion_use_prod_term,
            anchor_mode=fusion_anchor_mode,
            relation_only=fusion_relation_only,
            normalize_relation_terms=fusion_normalize_relation_terms,
            diff_mode=fusion_diff_mode,
        )
        self.multi_head = LiteStepMixHead(
            d_model=d_model,
            num_classes=num_classes,
            hidden_dim=2 * d_model,
            kernel_size=3,
            dropout=cross_dropout,
            pred_len=pred_len,
        )

    def bootstrap_multi_head_from_decoder(self):
        self.multi_head.bootstrap_from_decoder(self.decoder.fc_out)

    def sync_multi_decoder_from_shared(self):
        self.multi_decoder.load_state_dict(self.decoder.state_dict(), strict=True)

    def freeze_module(self, name):
        module = getattr(self, name)
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, numeric=None, image=None, mode="multi"):
        numeric_feature = self.numeric(numeric) if numeric is not None else None
        image_feature = self.vision(image) if image is not None else None

        if mode == "numeric":
            return self.decoder(numeric_feature, self.pred_len)
        if mode == "image":
            return self.decoder(image_feature, self.pred_len)
        if mode == "multi":
            numeric_hidden = self.multi_decoder.forward_hidden(numeric_feature, self.pred_len)
            image_hidden = self.multi_decoder.forward_hidden(image_feature, self.pred_len)
            fused_hidden = self.fusion(image_hidden, numeric_hidden)
            return self.multi_head(fused_hidden)
        raise ValueError("mode must be one of ['numeric', 'image', 'multi']")
