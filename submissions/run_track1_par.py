"""Track1 PAR code submission: pure-torch CLIP ViT-B/16 + attribute head.
Stdlib + torch + numpy + PIL only. No network. No transformers/safetensors.
Encoder: 12-layer ViT (768-wide, patch16) + visual projection + 2-layer head.
Weights: root-level part_XX.bin shards (see README); names embedded (no assets/ dependency).
"""
from __future__ import annotations
import json, os
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
MEAN = np.asarray([0.481, 0.457, 0.408], dtype=np.float32).reshape(3, 1, 1)
STD = np.asarray([0.268, 0.261, 0.275], dtype=np.float32).reshape(3, 1, 1)


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(768, 768, bias=True)
        self.k_proj = nn.Linear(768, 768, bias=True)
        self.v_proj = nn.Linear(768, 768, bias=True)
        self.out_proj = nn.Linear(768, 768, bias=True)
        self.num_heads = 12
        self.head_dim = 64
        self.scale = 64 ** -0.5

    def forward(self, x):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, 12, 64).transpose(1, 2)
        k = self.k_proj(x).view(b, s, 12, 64).transpose(1, 2)
        v = self.v_proj(x).view(b, s, 12, 64).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn.float(), dim=-1).to(x.dtype)
        o = torch.matmul(attn, v).transpose(1, 2).reshape(b, s, 768)
        return self.out_proj(o)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 3072, bias=True)
        self.fc2 = nn.Linear(3072, 768, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = x * torch.sigmoid(1.702 * x)
        return self.fc2(x)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()
        self.layer_norm1 = nn.LayerNorm(768, eps=1e-5)
        self.mlp = _MLP()
        self.layer_norm2 = nn.LayerNorm(768, eps=1e-5)

    def forward(self, x):
        r = x
        x = self.layer_norm1(x)
        x = self.self_attn(x)
        x = r + x
        r = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        return r + x


class _Embed(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Conv2d(3, 768, kernel_size=16, stride=16, bias=False)
        self.class_embedding = nn.Parameter(torch.zeros(768))
        self.position_embedding = nn.Embedding(197, 768)
        self.register_buffer("position_ids", torch.arange(197).expand((1, -1)), persistent=False)
        self.num_positions = 197

    def forward(self, pixel_values):
        p = self.patch_embedding(pixel_values)
        p = p.flatten(2).transpose(1, 2)
        cls = self.class_embedding.expand(p.shape[0], 1, -1)
        return (torch.cat([cls, p], dim=1) + self.position_embedding(self.position_ids))


class CLIPVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Module()
        self.vision.embeddings = _Embed()
        self.vision.pre_layrnorm = nn.LayerNorm(768, eps=1e-5)
        self.vision.encoder = nn.Module()
        self.vision.encoder.layers = nn.ModuleList([_Layer() for _ in range(12)])
        self.vision.post_layernorm = nn.LayerNorm(768, eps=1e-5)
        self.visual_projection = nn.Linear(768, 512, bias=False)

    def image_features(self, pixel_values):
        h = self.vision.embeddings(pixel_values)
        h = self.vision.pre_layrnorm(h)
        for layer in self.vision.encoder.layers:
            h = layer(h)
        pooled = h[:, 0, :]
        pooled = self.vision.post_layernorm(pooled)
        return self.visual_projection(pooled)


_MODEL = None
_DEVICE = None
_NAMES = ["Age-Young", "Age-Adult", "Age-Old", "Gender-Female", "Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald", "UpperBody-Length-Short", "UpperBody-Color-Black", "UpperBody-Color-Blue", "UpperBody-Color-Brown", "UpperBody-Color-Green", "UpperBody-Color-Grey", "UpperBody-Color-Orange", "UpperBody-Color-Pink", "UpperBody-Color-Purple", "UpperBody-Color-Red", "UpperBody-Color-White", "UpperBody-Color-Yellow", "UpperBody-Color-Other", "LowerBody-Length-Short", "LowerBody-Color-Black", "LowerBody-Color-Blue", "LowerBody-Color-Brown", "LowerBody-Color-Green", "LowerBody-Color-Grey", "LowerBody-Color-Orange", "LowerBody-Color-Pink", "LowerBody-Color-Purple", "LowerBody-Color-Red", "LowerBody-Color-White", "LowerBody-Color-Yellow", "LowerBody-Color-Other", "LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress", "Accessory-Backpack", "Accessory-Bag", "Accessory-Glasses-Normal", "Accessory-Glasses-Sun", "Accessory-Hat"]


def _load_weights():
    return _load_weights_impl()
