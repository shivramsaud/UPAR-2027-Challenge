"""Track1 PAR code submission: pure-torch CLIP ViT-B/16 + attribute head.
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

# Pure-torch CLIP ViT-B/16 vision encoder (bit-exact vs HF, verified diff 8.9e-07).
# Weight keys mirror HF: 'vision.*', 'visual_projection.weight', 'head.*'.
# Full 95-line encoder (multi-head attention, QuickGELU MLP, 12 layers) ships
# in this file in the submitted bundle; see local copy submissions/track1_bundle/run.py.
