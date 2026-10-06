"""Track3 pose submission: ViTPose TorchScript + UDP/DARK decode + flip-TTA.
See local submissions/track3_bundle/run.py for the full 250-line version;
entry points: load_model() once, predict_image(sample) -> {keypoints, scores}.
Weights: assets/vitpose.torchscript (+ root model.bin fallback)."""
from __future__ import annotations
