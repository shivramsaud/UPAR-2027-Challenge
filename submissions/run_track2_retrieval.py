"""Track2 retrieval code submission: pure-torch calibrated CLIP + weighted-LL ranking.
Stdlib + torch + numpy + PIL only. No network. No transformers/safetensors.
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


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
_W = None


def _load_weights():
    seen = {}
    try:
        seen["here"] = sorted(os.listdir(HERE))
    except Exception:
        seen["here"] = []
    try:
        seen["assets"] = sorted(os.listdir(ASSETS))
    except Exception:
        seen["assets"] = []
    sd = {}
    root_parts = sorted(f for f in seen["here"] if f.startswith("part_") and f.endswith(".bin"))
    if root_parts:
        for f in root_parts:
            sd.update(torch.load(os.path.join(HERE, f), map_location="cpu", weights_only=True))
        return sd
    for name in ("model_vision.bin", "model_vision.pth"):
        p = os.path.join(ASSETS, name)
        if os.path.isfile(p):
            return torch.load(p, map_location="cpu", weights_only=True)
    prefix = [f for f in seen["assets"] if f.startswith("shard_") and f.endswith((".pth", ".bin"))]
    if prefix:
        for f in sorted(prefix):
            sd.update(torch.load(os.path.join(ASSETS, f), map_location="cpu", weights_only=True))
        return sd
    raise FileNotFoundError("no weights found; HERE=%s ASSETS=%s" % (seen["here"], seen["assets"]))


def load_model():
    global _MODEL, _DEVICE, _NAMES, _W
    if _MODEL is not None:
        return
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    vision = CLIPVision()
    sd = _load_weights()
    vision.load_state_dict({k: sd[k] for k in vision.state_dict().keys()}, strict=True)
    head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 40))
    head.load_state_dict({k: sd["head." + k] for k in head.state_dict().keys()}, strict=True)
    temp = sd["temp"].clamp(0.3, 3).to(_DEVICE)
    bias = sd["bias"].to(_DEVICE)
    vision.eval()
    head.eval()
    _MODEL = (vision.to(_DEVICE), head.to(_DEVICE), temp, bias)
    freq = np.asarray([0.011, 0.965, 0.0237, 0.4292, 0.6546, 0.3114, 0.0126, 0.6288, 0.2556, 0.0821, 0.0199, 0.0435, 0.1621, 0.0058, 0.0105, 0.0253, 0.0694, 0.225, 0.0497, 0.0753, 0.3751, 0.4877, 0.1788, 0.0448, 0.015, 0.1719, 0.0021, 0.0155, 0.0008, 0.0039, 0.0447, 0.0017, 0.0372, 0.9038, 0.0958, 0.2086, 0.2931, 0.1394, 0.0218, 0.0425], dtype=np.float64).clip(0.05, 0.95)
    _W = (1.0 / np.sqrt(freq * (1.0 - freq))).astype(np.float32)


def _preprocess(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(((arr.transpose(2, 0, 1) - MEAN) / STD).astype(np.float32))


def predict_attributes(gallery, attribute_names):
    if _MODEL is None:
        load_model()
    vision, head, temp, bias = _MODEL
    lut = {n: i for i, n in enumerate(_NAMES)}
    norm = {str(n).strip().lower().replace("_", "-"): i for i, n in enumerate(_NAMES)}

    def _map(nm):
        if nm in lut:
            return lut[nm]
        k = str(nm).strip().lower().replace("_", "-")
        if k in norm:
            return norm[k]
        raise KeyError("unknown attribute %r" % nm)
    idx = [_map(n) for n in attribute_names]
    paths = []
    _roots = ["/app", "/data", os.getcwd(), "/app/data", "/app/input"]
    for gi, g in enumerate(gallery):
        p = None
        for k in ("image_path", "file_name", "path", "image", "filename", "img_path", "filepath", "file"):
            v = g.get(k) if isinstance(g, dict) else None
            if v and os.path.isfile(str(v)):
                p = str(v)
                break
            if v:
                bn = os.path.basename(str(v))
                for r in _roots:
                    c = os.path.join(r, str(v))
                    if os.path.isfile(c):
                        p = c
                        break
                    for cand in (os.path.join(r, bn), os.path.join(r, "gallery", bn), os.path.join(r, "images", bn)):
                        if os.path.isfile(cand):
                            p = cand
                            break
                    if p:
                        break
            if p:
                break
        if not p:
            raise FileNotFoundError("gallery[%d] unresolvable keys=%s" % (gi, list(g.keys()) if isinstance(g, dict) else type(g)))
        paths.append(p)
    out = []
    for i in range(0, len(paths), 32):
        batch = torch.stack([_preprocess(p) for p in paths[i:i + 32]]).to(_DEVICE)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=_DEVICE.type == "cuda"):
            logits = (head(vision.image_features(batch)) + bias) / temp
        out.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(out, axis=0)[:, np.asarray(idx)]


def rank_gallery(sample):
    if _MODEL is None:
        load_model()
    names = sample["attribute_names"]
    probs = predict_attributes(sample["gallery"], names).clip(1e-6, 1 - 1e-6)
    queries = np.asarray(sample["queries"], dtype=np.float32)
    if queries.ndim == 1:
        queries = queries[None, :]
    lut = {n: i for i, n in enumerate(_NAMES)}
    norm = {str(n).strip().lower().replace("_", "-"): i for i, n in enumerate(_NAMES)}

    def _map(nm):
        if nm in lut:
            return lut[nm]
        return norm[str(nm).strip().lower().replace("_", "-")]
    w = _W[[_map(n) for n in names]]
    ll = (queries[:, None, :] * np.log(probs)[None] + (1 - queries[:, None, :]) * np.log(1 - probs)[None]) * w
    return {"distances": (-ll.sum(-1)).astype(np.float64)}
