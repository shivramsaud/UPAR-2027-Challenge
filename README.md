# UPAR Challenge 2027 — Pedestrian Attribute Recognition, Retrieval & Pose Estimation

[![Kaggle](https://img.shields.io/badge/Kaggle-2xT4-GPU-blue)](https://www.kaggle.com)
[![HuggingFace](https://img.shields.io/badge/Models-HuggingFace-yellow)](https://huggingface.co/ShivRamSaud)
[![WACV 2027 RWS](https://img.shields.io/badge/WACV'27-Real--World_Surveillance-green)](https://vap.aau.dk/rws)

Competition entries + training code for all three tracks of the **UPAR Challenge 2027 @ Real-World Surveillance Workshop (WACV 2027)** — unified 40-attribute pedestrian recognition (Market1501 + PA-100K + PETA) and top-down 2D pose estimation under domain shift.

| Track | Task | Codabench | Model | Best | HF weights |
|---|---|---|---|---|---|
| 1 | Pedestrian Attribute Recognition | [18180](https://www.codabench.org/competitions/18180/) | CLIP ViT-B/16 → **ViT-L/14** | val-acc **0.9578** | [par-clip](https://huggingface.co/ShivRamSaud/upar-track1-par-clip) · [par-large](https://huggingface.co/ShivRamSaud/upar-track1-par-large) |
| 2 | Attribute-Based Person Retrieval | [18196](https://www.codabench.org/competitions/18196/) | Calibrated CLIP classifier + weighted-LL ranking | val-acc 0.9485 | [retrieval](https://huggingface.co/ShivRamSaud/upar-track2-retrieval) |
| 3 | 2D Human Pose Estimation | [18273](https://www.codabench.org/competitions/18273/) | ViTPose-S → **ViTPose-L** + flip-TTA | **0.8143** (was 0.2935) | [pose](https://huggingface.co/ShivRamSaud/upar-track3-pose) · [pose-large](https://huggingface.co/ShivRamSaud/upar-track3-pose-large) |

## Repo layout

```
tracks/track1_par/      base (ViT-B) + large (ViT-L/14) training notebooks, Kaggle output log, metrics
tracks/track2_retrieval/  training notebook, Kaggle output log, metrics
tracks/track3_pose/       base (ViTPose-S) + large (ViTPose-L) notebooks, Kaggle output log, metrics
submissions/            exact run.py entry points shipped to Codabench (code submissions)
```

Notebooks are the **actually-ran Kaggle versions**; `kaggle_output.log` files are the trimmed platform logs (progress lines only — no multi-MB dumps). Model weights (`.pth`/`.bin`/`.torchscript`/zips) live on HuggingFace (see table), not in git.

## Method (short)

- **T1:** CLIP image encoder + 2-layer attribute head, asymmetric multi-label loss, best-val checkpointing. Pure-torch inference (no `transformers` in the container).
- **T2:** same encoder, per-attribute temperature calibration, inverse-frequency weighted log-likelihood ranking → distances.
- **T3:** `usyd-community/vitpose-plus-{small,large}` (COCO-pretrained, rule-compliant), full real-image data, occlusion-aware aug, val-selected best, TorchScript export, flip-TTA at inference.

## Reproduce

```bash
pip install -r requirements.txt
# training ran on Kaggle 2xT4 (GPU) with HF-Hub checkpointing; see per-track notebooks
```

## Data

- Tracks 1–2: [UPAR-Challenge-2027](https://github.com/speckean/UPAR-Challenge-2027) (Market1501 + PA-100K + PETA, 40 attrs, 97k train)
- Track 3: [UPAR-Challenge-2027-Track3](https://github.com/MickaelCormier/UPAR-Challenge-2027-Track3)

## Citations

Specker et al., UPAR, WACV 2023 · Cormier et al., UPAR Challenge, WACVW 2023/2024 · Xu et al., ViTPose, NeurIPS 2022 / TPAMI 2024 · Wang et al., PromptPAR, TCSVT 2024.
