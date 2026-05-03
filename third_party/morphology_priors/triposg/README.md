# TripoSG — Image / Text-to-3D Mesh Prior

Empty checkout point for the main 3D morphology prior used in writeup §2.

## Setup (Phase 3)

```sh
cd third_party/morphology_priors/triposg
git clone https://github.com/VAST-AI-Research/TripoSG.git .
# follow upstream README:
#   pip install -r requirements.txt
#   download weights into checkpoints/
```

## Why TripoSG first

| Property | TripoSG | TRELLIS | Hunyuan3D-2 |
|---|---|---|---|
| Output | mesh | mesh + 3DGS + RF | mesh |
| Inference VRAM | ~12 GB | ~16 GB | ~24 GB |
| License | open | open | open |
| Adapter complexity | low | medium (structured latent decoding) | medium |

TripoSG has the smallest VRAM footprint and a simple `image → mesh` API,
making it the fastest path to a working asset bank. The other two go in as
priors-ablation in Table 3 once the pipeline is end-to-end.

## What lives here once cloned

- `triposg/`            ← upstream package
- `checkpoints/`        ← model weights (gitignored)
- adapter glue lives at `genedynamics/morphology/priors/triposg.py` (NOT here)
