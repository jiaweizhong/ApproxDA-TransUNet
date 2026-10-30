"""Swin-Unet (Cao et al., ECCV Workshops 2022) as in the official repository (HuCaoFighting/Swin-Unet, commit f48f623):
networks/vision_transformer.py with the hyper-parameters of configs/swin_tiny_patch4_window7_224_lite.yaml and the
config.py defaults written out (patch 4, embed 96, encoder depths [2, 2, 2, 2], heads [3, 6, 12, 24], window 7, MLP 4,
qkv bias, no APE, patch norm, drop rate 0, drop path 0.2, final upsample "expand_first").

As in the official wrapper, the yaml's DECODER_DEPTHS is not passed on, so the decoder uses SwinTransformerSys'
default [1, 2, 2, 2]. load_from() is the official ImageNet Swin-T loading (encoder weights, copied into the mirrored
decoder stages, shape-mismatched tensors dropped); the only change is torch.load(weights_only=True)."""
import copy

import torch
import torch.nn as nn

from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys


class SwinUnet(nn.Module):
    def __init__(self, img_size=224, num_classes=9):
        super().__init__()
        if img_size != 224:
            raise SystemExit("Swin-Unet (patch 4, window 7, 3 patch mergings) needs 224 x 224 input")
        self.num_classes = num_classes
        self.swin_unet = SwinTransformerSys(img_size=img_size, patch_size=4, in_chans=3, num_classes=num_classes,
                                            embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24], window_size=7,
                                            mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0,
                                            drop_path_rate=0.2, ape=False, patch_norm=True, use_checkpoint=False)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.swin_unet(x)

    def load_from(self, pretrained_path):
        pretrained_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)["model"]
        model_dict = self.swin_unet.state_dict()
        full_dict = copy.deepcopy(pretrained_dict)
        for k, v in pretrained_dict.items():
            if "layers." in k:
                full_dict.update({"layers_up." + str(3 - int(k[7:8])) + k[8:]: v})
        for k in list(full_dict.keys()):
            if k in model_dict and full_dict[k].shape != model_dict[k].shape:
                del full_dict[k]
        msg = self.swin_unet.load_state_dict(full_dict, strict=False)
        print(f"loaded Swin-T ImageNet weights from {pretrained_path}: {len(full_dict)} tensors, "
              f"{len(msg.missing_keys)} model tensors left at init")
        return msg
