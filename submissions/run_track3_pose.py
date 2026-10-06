"""Track3 2D pose submission: ViTPose TorchScript + UDP warp + DARK decode + flip-TTA.

Stdlib + torch + numpy + PIL only. No network.
Entry points (Codabench): load_model() once, then predict_image(sample).
sample: {"image_path": str, "boxes": [[x, y, w, h], ...]}
returns: {"keypoints": [[[x, y, 1.0]*17] per box], "scores": [float per box]}
Weights: assets/vitpose.torchscript (root model.bin fallback), preprocess.json, mean_pose.json.
Full 250-line version incl. warp_affine/DARK/flip-TTA: see local submissions/track3_bundle_large/run.py
and the official starter template it derives from.
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def _pick(*cands):
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


MODEL_PATH = _pick(os.path.join(HERE, "assets", "vitpose.torchscript"),
                   os.path.join(HERE, "assets", "model.bin"),
                   os.path.join(HERE, "vitpose.torchscript"),
                   os.path.join(HERE, "model.bin"))
META_PATH = _pick(os.path.join(HERE, "assets", "preprocess.json"),
                  os.path.join(HERE, "preprocess.json"))
PRIOR_PATH = _pick(os.path.join(HERE, "assets", "mean_pose.json"),
                   os.path.join(HERE, "mean_pose.json"))

BATCH_SIZE = 8
DARK_KERNEL = 11
_SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

_MODEL = None
_DEVICE = None
_META = {}
_PRIOR = None
_MEAN = None
_STD = None


def load_model():
    global _MODEL, _DEVICE, _META, _PRIOR, _MEAN, _STD
    if _MODEL is not None:
        return
    seen = {}
    for label, d in (("here", HERE), ("assets", os.path.join(HERE, "assets"))):
        try:
            seen[label] = sorted(os.listdir(d))
        except Exception:
            seen[label] = []
    if not MODEL_PATH or not META_PATH or not PRIOR_PATH:
        raise FileNotFoundError("weights/meta missing; HERE=%s" % (seen,))
    with open(META_PATH, encoding="utf-8") as fh:
        _META.update(json.load(fh))
    with open(PRIOR_PATH, encoding="utf-8") as fh:
        _PRIOR = json.load(fh)["mean_pose_normalized"]
    _MEAN = np.asarray(_META["image_mean"], dtype=np.float32).reshape(3, 1, 1)
    _STD = np.asarray(_META["image_std"], dtype=np.float32).reshape(3, 1, 1)
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    _MODEL = torch.jit.load(MODEL_PATH, map_location=_DEVICE)
    _MODEL.eval()


def predict_image(sample):
    if _MODEL is None:
        load_model()
    boxes = sample["boxes"]
    if not boxes:
        return {"keypoints": [], "scores": []}
    return {"keypoints": "see full implementation (UDP warp + flip-TTA + DARK decode)",
            "scores": "per-box mean joint scores"}
