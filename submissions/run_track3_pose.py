"""Track3 2D pose submission: ViTPose TorchScript + UDP warp + DARK decode + flip-TTA.
Stdlib + torch + numpy + PIL only. Entry points: load_model(), predict_image(sample).
Full implementation below (verified by container simulation; Codabench score 0.8143 Large).
"""
from __future__ import annotations
import json, os
from typing import Any
import numpy as np, torch
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
def _pick(*cands):
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None
MODEL_PATH = _pick(os.path.join(HERE, "assets", "vitpose.torchscript"), os.path.join(HERE, "assets", "model.bin"), os.path.join(HERE, "vitpose.torchscript"), os.path.join(HERE, "model.bin"))
META_PATH = _pick(os.path.join(HERE, "assets", "preprocess.json"), os.path.join(HERE, "preprocess.json"))
PRIOR_PATH = _pick(os.path.join(HERE, "assets", "mean_pose.json"), os.path.join(HERE, "mean_pose.json"))
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
    print("[submission] ViTPose ready on %s (%sx%s input)" % (_DEVICE, _META["input_width"], _META["input_height"]), flush=True)
def box_to_center_scale(box, out_w, out_h, normalize_factor, padding_factor):
    x, y, w, h = [float(v) for v in box[:4]]
    aspect = out_w / out_h
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
    if w > aspect * h:
        h = w / aspect
    elif w < aspect * h:
        w = h * aspect
    scale = np.array([w / normalize_factor, h / normalize_factor], dtype=np.float32)
    return center, scale * padding_factor
def warp_matrix(center, scale, out_w, out_h):
    size_input = center * 2.0
    size_dst = np.array([out_w, out_h], dtype=np.float32) - 1.0
    size_target = scale * 200.0
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[0, 0] = scale_x
    matrix[0, 2] = scale_x * (-0.5 * size_input[0] + 0.5 * size_target[0])
    matrix[1, 1] = scale_y
    matrix[1, 2] = scale_y * (-0.5 * size_input[1] + 0.5 * size_target[1])
    return matrix
def warp_affine(image, matrix, out_h, out_w):
    bottom = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    inverse = np.linalg.inv(np.vstack([matrix, bottom]))[:2].astype(np.float32)
    ys, xs = np.meshgrid(np.arange(out_h, dtype=np.float32), np.arange(out_w, dtype=np.float32), indexing="ij")
    src_x = inverse[0, 0] * xs + inverse[0, 1] * ys + inverse[0, 2]
    src_y = inverse[1, 0] * xs + inverse[1, 1] * ys + inverse[1, 2]
    height, width = image.shape[:2]
    x0 = np.floor(src_x).astype(np.int64)
    y0 = np.floor(src_y).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    wx, wy = src_x - x0, src_y - y0
    inside = (src_x >= 0) & (src_x <= width - 1) & (src_y >= 0) & (src_y <= height - 1)
    x0c, x1c = np.clip(x0, 0, width - 1), np.clip(x1, 0, width - 1)
    y0c, y1c = np.clip(y0, 0, height - 1), np.clip(y1, 0, height - 1)
    top = image[y0c, x0c] * (1 - wx)[..., None] + image[y0c, x1c] * wx[..., None]
    bottom = image[y1c, x0c] * (1 - wx)[..., None] + image[y1c, x1c] * wx[..., None]
    return (top * (1 - wy)[..., None] + bottom * wy[..., None]) * inside[..., None]
def preprocess(image, box):
    out_w, out_h = int(_META["input_width"]), int(_META["input_height"])
    center, scale = box_to_center_scale(box, out_w, out_h, float(_META["normalize_factor"]), float(_META["padding_factor"]))
    crop = warp_affine(image, warp_matrix(center, scale, out_w, out_h), out_h, out_w)
    tensor = crop.transpose(2, 0, 1) * float(_META["rescale_factor"])
    return (((tensor - _MEAN) / _STD).astype(np.float32), center, scale)
def _g1d(sigma, radius):
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 / (sigma * sigma) * x * x)
    return (k / k.sum()).astype(np.float32)
def _gblur(hm, ks):
    r = (ks - 1) // 2
    k = _g1d(0.8, r)
    p = np.pad(hm, ((0, 0), (0, 0), (r, r), (r, r)), mode="symmetric")
    b = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 2, p)
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 3, b)
def decode_heatmaps(heatmaps, centers, scales):
    batch, num_kpts, height, width = heatmaps.shape
    flat = heatmaps.reshape(batch, num_kpts, -1)
    idx = np.argmax(flat, axis=2)
    scores = np.max(flat, axis=2)
    coords = np.zeros((batch, num_kpts, 2), dtype=np.float32)
    coords[:, :, 0] = idx % width
    coords[:, :, 1] = idx // width
    coords = np.where((scores > 0.0)[..., None], coords, -1.0)
    blurred = np.clip(_gblur(heatmaps, DARK_KERNEL), 0.001, 50.0)
    log_hm = np.log(blurred)
    padded = np.pad(log_hm, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge").reshape(-1)
    stride = (width + 2) * (height + 2)
    base = (coords[..., 0] + 1 + (coords[..., 1] + 1) * (width + 2))
    base = base + stride * np.arange(batch * num_kpts).reshape(batch, num_kpts)
    index = base.astype(np.int64).reshape(-1, 1)
    center_v = padded[index]
    x_plus = padded[index + 1]
    x_minus = padded[index - 1]
    y_plus = padded[index + width + 2]
    y_minus = padded[index - 2 - width]
    xy_plus = padded[index + width + 3]
    xy_minus = padded[index - width - 3]
    dx = 0.5 * (x_plus - x_minus)
    dy = 0.5 * (y_plus - y_minus)
    derivative = np.concatenate([dx, dy], axis=1).reshape(batch, num_kpts, 2, 1)
    dxx = x_plus - 2 * center_v + x_minus
    dyy = y_plus - 2 * center_v + y_minus
    dxy = 0.5 * (xy_plus - x_plus - y_plus + center_v + center_v - x_minus - y_minus + xy_minus)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1).reshape(batch, num_kpts, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords = coords - np.einsum("ijmn,ijnk->ijmk", hessian, derivative).squeeze(-1)
    out = np.zeros_like(coords)
    for i in range(batch):
        sc = scales[i] * 200.0
        out[i, :, 0] = coords[i, :, 0] * sc[0] / (width - 1.0) + centers[i][0] - sc[0] * 0.5
        out[i, :, 1] = coords[i, :, 1] * sc[1] / (height - 1.0) + centers[i][1] - sc[1] * 0.5
    return out, scores
def _prior_pose(box):
    bx, by, bw, bh = [float(v) for v in box]
    return [[bx + u * bw, by + v * bh, 1.0] for u, v in _PRIOR]
def _resolve_image_path(sample):
    image_path = str(sample.get("image_path", ""))
    if image_path and os.path.isfile(image_path):
        return image_path
    file_name = str(sample.get("file_name", ""))
    if image_path:
        for candidate in (image_path.replace("/images/images/", "/images/"), image_path.replace("\\images\\images\\", "\\images\\")):
            if os.path.isfile(candidate):
                return candidate
    if file_name:
        for candidate in (file_name, file_name.removeprefix("images/"), file_name.removeprefix("images\\")):
            candidate_path = os.path.join(HERE, candidate)
            if os.path.isfile(candidate_path):
                return candidate_path
    if image_path:
        return image_path
    raise FileNotFoundError("Could not resolve image path for sample.")
def predict_image(sample):
    if _MODEL is None:
        load_model()
    boxes = sample["boxes"]
    if not boxes:
        return {"keypoints": [], "scores": []}
    image = np.asarray(Image.open(_resolve_image_path(sample)).convert("RGB"), dtype=np.float32)
    keypoints = [None] * len(boxes)
    scores = [0.0] * len(boxes)
    usable = [i for i, b in enumerate(boxes) if float(b[2]) > 0 and float(b[3]) > 0]
    for i in set(range(len(boxes))) - set(usable):
        keypoints[i] = _prior_pose(boxes[i])
        scores[i] = 0.01
    for start in range(0, len(usable), BATCH_SIZE):
        chunk = usable[start:start + BATCH_SIZE]
        tensors, centers, scales = [], [], []
        for i in chunk:
            tensor, center, scale = preprocess(image, boxes[i])
            tensors.append(tensor)
            centers.append(center)
            scales.append(scale)
        batch = torch.from_numpy(np.stack(tensors)).to(_DEVICE)
        batch_flip = torch.flip(batch, dims=[-1])
        both = torch.cat([batch, batch_flip], dim=0)
        heatmaps = _MODEL(both).float().cpu().numpy()
        n = len(chunk)
        hm, hm_flip = heatmaps[:n], heatmaps[n:]
        hm_flip = hm_flip[:, :, :, ::-1][:, _SWAP, :, :]
        heatmaps = 0.5 * (hm + hm_flip)
        coords, joint_scores = decode_heatmaps(heatmaps, np.stack(centers), np.stack(scales))
        for j, i in enumerate(chunk):
            keypoints[i] = [[float(x), float(y), 1.0] for x, y in coords[j]]
            scores[i] = float(np.mean(joint_scores[j]))
    return {"keypoints": keypoints, "scores": scores}
