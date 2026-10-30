# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import configs as configs
from .block import ResNetV2
from .block import ApproxDABlock, ApproxDABlockLegacy, MedNeXtBlock, RefineBlock
from .lightvit import lightvit_tiny, lightvit_small, lightvit_base, ResidualMergePatch


def make_approx_block(channels, config, skip=False):
    """Build the attention block selected by config.block_version ('c16' | 'legacy').

    skip=True for the blocks on the skip connections: their C/r width follows config.skip_ratio (default 16).
    """
    version = getattr(config, "block_version", "c16")
    if getattr(config, "no_block", False):  # journal diagnostic: no attention blocks at all (TransUNet-style)
        return nn.Identity()
    if getattr(config, "block_type", "approxda") == "refine":  # attention-free DANetHead conv path (RefineBlock)
        mode = getattr(config, "refine_mode", "plain")
        if mode == "guided" and not skip:
            mode = "plain"  # the bottleneck block has no decoder feature to be guided by
        if mode == "guided" and getattr(config, "dec_channels", 0) <= 0:
            raise SystemExit("--refine_mode guided requires the slim decoder (--dec_channels > 0)")
        return RefineBlock(channels, ratio=getattr(config, "skip_ratio", 16) if skip else 16, refine_mode=mode,
                           guide_channels=config.dec_channels if mode == "guided" else 0)
    if getattr(config, "block_type", "approxda") == "mednext":  # attention-free conv block instead of ApproxDA
        return MedNeXtBlock(channels, kernel_size=getattr(config, "mednext_k", 5), exp=getattr(config, "mednext_exp", 2))
    cls = {"c16": ApproxDABlock, "legacy": ApproxDABlockLegacy}[version]
    extra = {}
    if version == "c16":  # PAM scale options (defaults reproduce the reported runs)
        extra = dict(alpha_init=getattr(config, "alpha_init", 0.0), attn_norm=getattr(config, "attn_norm", False),
                     pam_identity=not getattr(config, "pam_no_identity", False),
                     pam_style=getattr(config, "pam_style", "danet"), pam_fusion=getattr(config, "pam_fusion", "residual"),
                     attn_off=getattr(config, "attn_off", False)
                     and getattr(config, "attn_off_at", "all") in ("all", "skip" if skip else "bottleneck"),
                     ratio=getattr(config, "skip_ratio", 16) if skip else 16,
                     pam_relpos=getattr(config, "pam_relpos", False), pam_abspos=getattr(config, "pam_abspos", False),
                     coord_gate=getattr(config, "coord_gate", False),
                     # --pam_affinity: Context Prior affinity supervision on the bottleneck PAM only (global window)
                     pam_affinity=getattr(config, "pam_affinity", 0) > 0 and not skip,
                     # --third_branch organ|triplet: third branch of the bottleneck block only
                     third_branch="none" if skip else getattr(config, "third_branch", "none"),
                     num_classes=getattr(config, "n_classes", 9))
    return cls(
        channels,
        window_size=config.window_size,
        rank=config.rank,
        groups=config.groups,
        gate_mode=config.gate_mode,
        **extra,
    )


# Skip channels that get an attention block, and the attribute each is stored
# under (names kept from the original da/da2/da3 layout).
_SKIP_BLOCK_NAMES = {64: "da", 256: "da2", 512: "da3"}

from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    ReLU,
    AdaptiveMaxPool2d,
    AdaptiveAvgPool2d,
    NLLLoss,
    BCELoss,
    CrossEntropyLoss,
    AvgPool2d,
    MaxPool2d,
    Parameter,
    Linear,
    Sigmoid,
    Softmax,
    Dropout,
    Embedding,
)
from torch.nn import functional as F
from torch.autograd import Variable

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {
    "gelu": torch.nn.functional.gelu,
    "relu": torch.nn.functional.relu,
    "swish": swish,
}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings."""

    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        self.approx_block = make_approx_block(config.hidden_size, config)

        if config.patches.get("grid") is not None:  # ResNet
            grid_size = config.patches["grid"]
            patch_size = (
                img_size[0] // 16 // grid_size[0],
                img_size[1] // 16 // grid_size[1],
            )
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (
                img_size[1] // patch_size_real[1]
            )
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(
                block_units=config.resnet.num_layers,
                width_factor=config.resnet.width_factor,
            )
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches, config.hidden_size)
        )

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = self.approx_block(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2).contiguous()  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = (
                np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )
            key_weight = (
                np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )
            value_weight = (
                np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )
            out_weight = (
                np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(
                np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")])
            )
            self.attention_norm.bias.copy_(
                np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")])
            )
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels=0,
        use_batchnorm=True,
        config=None,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        # Build only the block this level's skip uses (previously all three were
        # built in every DecoderBlock, leaving 9 never-called blocks).
        self.block_name = _SKIP_BLOCK_NAMES.get(skip_channels)
        if self.block_name is not None:
            setattr(self, self.block_name, make_approx_block(skip_channels, config, skip=True))

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            skip = skip.contiguous()
            if self.block_name is not None:
                skip = getattr(self, self.block_name)(skip)

            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(
                4 - self.config.n_skip
            ):  # re-select the skip channels according to n_skip
                skip_channels[3 - i] = 0

        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(
                in_ch,
                out_ch,
                sk_ch,
                config=config,
            )
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = (
            hidden_states.size()
        )  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


# ---------------------------------------------------------------------------
# Journal experiments (opt-in; defaults build the original model unchanged)
# ---------------------------------------------------------------------------

_LIGHTVIT = {"lightvit_t": lightvit_tiny, "lightvit_s": lightvit_small, "lightvit_b": lightvit_base}


class LightViTBottleneck(nn.Module):
    """--backbone lightvit_{t,s,b}: replaces R50 block3 and the 12-layer ViT-B.

    R50's 1/8 feature map is projected to LightViT's stage-1 width and passed through LightViT's three stages
    (strides 8/16/32 of the input, ImageNet-1k weights incl. patch merging and global tokens); the stage-2 map
    (1/16) and the upsampled stage-3 map (1/32) are fused into the 1/16 bottleneck map used by the decoder.
    """

    def __init__(self, name, in_channels):
        super().__init__()
        lv = _LIGHTVIT[name]()
        self.num_tokens = lv.num_tokens
        d1, d2, d3 = lv.embed_dims
        self.in_proj = nn.Conv2d(in_channels, d1, kernel_size=1)
        self.in_norm = nn.LayerNorm(d1)
        self.global_token = lv.global_token
        self.stages = lv.stages
        self.norm2 = nn.LayerNorm(d2)
        self.norm3 = nn.LayerNorm(d3)
        self.fuse = nn.Conv2d(d3, d2, kernel_size=1)
        self.out_channels = d2

    def load_pretrained(self, path):
        sd = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
        own = {k: v for k, v in sd.items() if k == "global_token" or k.startswith("stages.")}
        missing, unexpected = self.load_state_dict(own, strict=False)
        missing = [k for k in missing if k == "global_token" or k.startswith("stages.")]
        if missing or unexpected:
            raise RuntimeError(f"LightViT weights mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
        return len(own)

    def forward(self, f):
        B, _, H, W = f.shape
        x = self.in_norm(self.in_proj(f).flatten(2).transpose(1, 2))
        x = torch.cat([self.global_token.expand(B, -1, -1), x], dim=1)
        outs = []
        for stage in self.stages:
            for block in stage:
                if isinstance(block, ResidualMergePatch):
                    x, (H, W) = block(x, H, W)
                else:
                    x = block(x, H, W)
            outs.append((x[:, self.num_tokens:], H, W))
        (x2, H2, W2), (x3, H3, W3) = outs[1], outs[2]
        m2 = self.norm2(x2).transpose(1, 2).reshape(B, -1, H2, W2)
        m3 = self.norm3(x3).transpose(1, 2).reshape(B, -1, H3, W3)
        return m2 + F.interpolate(self.fuse(m3), size=(H2, W2), mode="bilinear", align_corners=False)


class ConvBottleneck(nn.Module):
    """--backbone conv (additive-ablation baseline): the transformer-free stand-in for LightViTBottleneck, with the same
    input (R50's 1/8 map) and output (the 1/16 map at LightViT-T's width 128): one 3x3 stride-2 conv-BN-ReLU."""

    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.out_channels = out_channels

    def forward(self, f):
        return self.body(f)


class LightViTTransformer(nn.Module):
    """Encoder for --backbone lightvit_*: R50 root/block1/block2 (skips at 1/2, 1/4, 1/8) + LightViTBottleneck.
    Returns the same (tokens, attn, features) triple as Transformer; config.hidden_size becomes the LightViT width."""

    def __init__(self, config):
        super().__init__()
        self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
        del self.hybrid_model.body.block3  # the 1/16 stage is replaced by LightViT
        self.lightvit = (ConvBottleneck(self.hybrid_model.width * 8) if config.backbone == "conv"
                         else LightViTBottleneck(config.backbone, self.hybrid_model.width * 8))
        config.hidden_size = self.lightvit.out_channels
        # bn_order (opt-in --bn_order pre): "post" (default, all reported light-tier runs) applies the bottleneck
        # ApproxDA block to the LightViT output (14^2 x 128); "pre" applies it to R50's 1/8 map (28^2 x 512) before
        # LightViT, the order of the ViT-B model (patch embedding -> ApproxDA block -> ViT, as DA-TransUNet)
        self.bn_order = getattr(config, "bn_order", "post")
        self.approx_block = make_approx_block(self.hybrid_model.width * 8 if self.bn_order == "pre" else config.hidden_size, config)

    def forward(self, x):
        r = self.hybrid_model
        b, _, in_size, _ = x.size()
        features = []
        x = r.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(r.body)):  # block1 (1/4), block2 (1/8); same padding rule as ResNetV2.forward
            x = r.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0 : x.size()[2], 0 : x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        if self.bn_order == "pre":
            y = self.lightvit(self.approx_block(features[-1]))
        else:
            y = self.approx_block(self.lightvit(features[-1]))
        return y.flatten(2).transpose(1, 2), None, features[::-1]


class ReduceBlock(nn.Module):
    """EffiDec-style channel-reduction residual block (2-D, BatchNorm): two 3x3 conv-BN(-ReLU) + 1x1 shortcut."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
        )
        self.res = (nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
                    if cin != cout else nn.Identity())
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.res(x))


class SlimDecoderBlock(nn.Module):
    """Upsample, add the ApproxDA-processed skip reduced to `channels`, refine."""

    def __init__(self, channels, skip_channels, config):
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.block_name = _SKIP_BLOCK_NAMES.get(skip_channels)
        if self.block_name is not None:
            setattr(self, self.block_name, make_approx_block(skip_channels, config, skip=True))
        self.reduce = ReduceBlock(skip_channels, channels) if skip_channels else None
        self.refine = ReduceBlock(channels, channels)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if self.block_name is not None:
                blk = getattr(self, self.block_name)
                if getattr(blk, "refine_mode", None) == "guided":
                    skip = blk(skip.contiguous(), guide=x)
                else:
                    skip = blk(skip.contiguous())
            x = x + self.reduce(skip)
        return self.refine(x)


class SlimDecoderCup(nn.Module):
    """--dec_channels C (EffiDec3D, adapted to 2-D): every skip stage at C channels with additive fusion.
    The full-resolution stage keeps the original 16 channels (at C channels it would cost more than the original),
    or is removed with --dec_stop112 (prediction at 1/2 resolution, upsampled x2 by the segmentation head)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        C = config.dec_channels
        self.conv_more = Conv2dReLU(config.hidden_size, C, kernel_size=3, padding=1, use_batchnorm=True)
        skip_channels = list(config.skip_channels[:3])
        for i in range(3 - config.n_skip):
            skip_channels[2 - i] = 0
        self.blocks = nn.ModuleList([SlimDecoderBlock(C, sk, config) for sk in skip_channels])
        # --final_channels F (default 16) and --hr_skip H (default 0, opt-in): the full-resolution stage concatenates the
        # upsampled decoder map with an H-channel feature computed from the input image at full resolution (as the top
        # skip of U-Net). The default path builds exactly the original modules (same parameter names).
        F_out = getattr(config, "final_channels", 16)
        hr = getattr(config, "hr_skip", 0)
        self.final_hr = None
        if hr > 0:
            if config.dec_stop112:
                raise ValueError("--hr_skip needs the full-resolution decoder stage (no --dec_stop112)")
            self.final = None
            self.final_up = nn.UpsamplingBilinear2d(scale_factor=2)
            self.final_hr = nn.Sequential(
                Conv2dReLU(C + hr, F_out, kernel_size=3, padding=1, use_batchnorm=True),
                Conv2dReLU(F_out, F_out, kernel_size=3, padding=1, use_batchnorm=True),
            )
        else:
            self.final = None if config.dec_stop112 else nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                Conv2dReLU(C, F_out, kernel_size=3, padding=1, use_batchnorm=True),
                Conv2dReLU(F_out, F_out, kernel_size=3, padding=1, use_batchnorm=True),
            )
        self.out_channels = C if config.dec_stop112 else F_out

    def forward(self, hidden_states, features=None, hr_feat=None):
        B, n_patch, hidden = hidden_states.size()
        h = w = int(np.sqrt(n_patch))
        x = self.conv_more(hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, h, w))
        for i, block in enumerate(self.blocks):
            skip = features[i] if (features is not None and i < self.config.n_skip) else None
            x = block(x, skip=skip)
        if self.final_hr is not None:
            return self.final_hr(torch.cat([self.final_up(x), hr_feat], dim=1))
        return x if self.final is None else self.final(x)


class ApproxDATransUNet(nn.Module):
    def __init__(
        self, config, img_size=224, num_classes=21843, zero_head=False, vis=False
    ):
        super(ApproxDATransUNet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.backbone = getattr(config, "backbone", "vit")
        if self.backbone != "vit":  # LightViTTransformer changes hidden_size: work on a copy of the shared config
            config = copy.deepcopy(config)
            self.transformer = LightViTTransformer(config)
        else:
            self.transformer = Transformer(config, img_size, vis)
        if getattr(config, "dec_channels", 0) > 0:
            self.decoder = SlimDecoderCup(config)
            self.segmentation_head = SegmentationHead(
                in_channels=self.decoder.out_channels,
                out_channels=config["n_classes"],
                kernel_size=3,
                upsampling=2 if config.dec_stop112 else 1,
            )
        else:
            self.decoder = DecoderCup(config)
            self.segmentation_head = SegmentationHead(
                in_channels=config["decoder_channels"][-1],
                out_channels=config["n_classes"],
                kernel_size=3,
            )
        self.config = config
        # --slice_attn K >= 0 (opt-in): inter-slice attention over the bottleneck tokens of 2K+1 adjacent slices
        K = getattr(config, "slice_attn", -1)
        self.slice_attn = None
        if K >= 0:
            from .block import SliceAttention

            self.slice_attn = SliceAttention(config.hidden_size, 2 * K + 1, (img_size // 16) ** 2)
        # --point_refine (opt-in): PointRend-style native-resolution boundary refinement head; forward keeps the
        # decoder output in self.last_dec_feat for it
        self.point_head = None
        if getattr(config, "point_refine", False):
            if K >= 0:
                raise ValueError("--point_refine is not implemented together with --slice_attn")
            from .block import PointRefineHead

            feat_ch = self.decoder.out_channels if getattr(config, "dec_channels", 0) > 0 else config["decoder_channels"][-1]
            self.point_head = PointRefineHead(config["n_classes"], feat_ch, residual=getattr(config, "pr_residual", False),
                                              balanced=getattr(config, "pr_balanced", False))
            self.last_dec_feat = None
        # --hr_skip H (opt-in, slim decoder only): two 3x3 conv-BN-ReLU on the (3-channel) input at full resolution,
        # fed to the decoder's full-resolution stage
        self.hr_stem = None
        if getattr(config, "hr_skip", 0) > 0:
            if getattr(config, "dec_channels", 0) <= 0 or K >= 0:
                raise ValueError("--hr_skip needs --dec_channels (slim decoder) and no --slice_attn")
            H = config.hr_skip
            self.hr_stem = nn.Sequential(Conv2dReLU(3, H, kernel_size=3, padding=1, use_batchnorm=True),
                                         Conv2dReLU(H, H, kernel_size=3, padding=1, use_batchnorm=True))

    def forward(self, x):
        if self.slice_attn is not None:
            return self._forward_slices(x)
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        hr_feat = self.hr_stem(x) if self.hr_stem is not None else None
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features, hr_feat) if hr_feat is not None else self.decoder(x, features)
        if self.point_head is not None:
            self.last_dec_feat = x
        logits = self.segmentation_head(x)
        return logits

    def _forward_slices(self, x):
        """--slice_attn: x (B, S, H, W) holds the 2K+1 adjacent grayscale slices, the centre one in the middle. All
        slices go through the shared encoder; the centre slice's bottleneck tokens attend to all of them; the decoder
        uses the centre slice's tokens and skips."""
        B, S = x.shape[:2]
        if S == 1 and self.slice_attn.num_slices > 1:  # single-slice dummy input (FLOP profiling)
            x, S = x.repeat(1, self.slice_attn.num_slices, 1, 1), self.slice_attn.num_slices
        tokens, _, features = self.transformer(x.reshape(B * S, 1, *x.shape[2:]).repeat(1, 3, 1, 1))
        tokens = self.slice_attn(tokens.view(B, S, *tokens.shape[1:]))
        features = [f.view(B, S, *f.shape[1:])[:, S // 2] for f in features]
        return self.segmentation_head(self.decoder(tokens, features))

    def load_from(self, weights):
        if self.backbone != "vit":
            return self._load_from_lightvit(weights)
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(
                np2th(weights["embedding/kernel"], conv=True)
            )
            self.transformer.embeddings.patch_embeddings.bias.copy_(
                np2th(weights["embedding/bias"])
            )

            self.transformer.encoder.encoder_norm.weight.copy_(
                np2th(weights["Transformer/encoder_norm/scale"])
            )
            self.transformer.encoder.encoder_norm.bias.copy_(
                np2th(weights["Transformer/encoder_norm/bias"])
            )

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info(
                    "load_pretrained: resized variant: %s to %s"
                    % (posemb.size(), posemb_new.size())
                )
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print("load_pretrained: grid-size from %s to %s" % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(res_weight["conv_root/kernel"], conv=True)
                )
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for (
                    bname,
                    block,
                ) in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

    def _load_from_lightvit(self, weights):
        """R50 root/block1/block2 from the R50+ViT-B_16 ImageNet-21k checkpoint, LightViT stages from ImageNet-1k."""
        r = self.transformer.hybrid_model
        with torch.no_grad():
            r.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
            r.root.gn.weight.copy_(np2th(weights["gn_root/scale"]).view(-1))
            r.root.gn.bias.copy_(np2th(weights["gn_root/bias"]).view(-1))
            for bname, block in r.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=bname, n_unit=uname)
        if self.backbone == "conv":  # transformer-free baseline: R50 weights only, the stand-in conv starts random
            print("loaded R50 root/block1/block2 (no LightViT; --backbone conv)")
            return
        n = self.transformer.lightvit.load_pretrained(self.config.lightvit_ckpt)
        print(f"loaded R50 root/block1/block2 and {n} LightViT tensors from {self.config.lightvit_ckpt}")


CONFIGS = {
    "ViT-B_16": configs.get_b16_config(),
    "ViT-B_32": configs.get_b32_config(),
    "ViT-L_16": configs.get_l16_config(),
    "ViT-L_32": configs.get_l32_config(),
    "ViT-H_14": configs.get_h14_config(),
    "R50-ViT-B_16": configs.get_r50_b16_config(),
    "R50-ViT-L_16": configs.get_r50_l16_config(),
    "testing": configs.get_testing(),
}
