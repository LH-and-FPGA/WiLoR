# Distilling the WiLoR ViT-H backbone

Halve the backbone's depth (32 → 16 layers) while keeping the rest of the pipeline
byte-for-byte unchanged.

## Why depth and not width

`wilor/models/wilor.py:39` hardcodes `RefineNet(cfg, feat_dim=1280)`, and RefineNet
consumes a `1280×16×12` feature map with `out_dim = 1280//8 + 1280//4 + 1280//2`.
Narrowing the ViT (e.g. to ViT-B's 768) therefore forces RefineNet to be retrained
too, which drags in the whole MANO supervision chain.

Depth is free. A 16-layer / 1280-wide student is a **drop-in replacement for
`model.backbone`**; RefineNet and both MANO passes stay frozen and shared. That also
buys two things:

- the student is **initialised from the teacher's own layers** (uniform stride, so
  student block *k* corresponds to teacher block `keep[k]`), not from scratch;
- intermediate hidden states can be matched layer-by-layer, which is what makes this
  converge in hours rather than days.

## No labels needed

Because everything downstream of the ViT is frozen and shared, a student whose ViT
outputs match the teacher's reproduces the entire pipeline. The teacher generates
every target, so **any collection of hand images works** — no ground truth required.

## Loss

| term | matches | why |
|---|---|---|
| `feat` | `img_feat` (1280×16×12) | RefineNet's input |
| `tok` | `pred_mano_feats` (pose 96, betas 10) | the readout tokens |
| `cam` | the camera token | weighted heavily: depth is `z = 2f/(bbox·s)`, so `s` is the fragile quantity |
| `hidden` | student block *k* ↔ teacher block `keep[k]` | fast convergence from a teacher init |
| `kp_rel` | 21 joints after frozen RefineNet + MANO | **the task-space term** |

`kp_rel` exists because the downstream multi-view fusion is 2D-limited: it
triangulates `landmarks_px` and touches the monocular 3D only through bone lengths
and a scale-invariant Kabsch fit. Optimising the final joints directly beats hoping
feature matching implies good landmarks.

## Pipeline

```bash
# 1. images (FreiHAND: 130 240 images, ~3.7 GB; any hand imagery works)
mkdir -p distill/data/downloads
curl -L -o distill/data/downloads/FreiHAND_pub_v2.zip \
  https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip
unzip -q distill/data/downloads/FreiHAND_pub_v2.zip 'training/rgb/*' -d distill/data

# 2. cache detector boxes once (~7 min; 130 223/130 240 kept)
CUDA_VISIBLE_DEVICES=0 PYGLET_HEADLESS=true PYOPENGL_PLATFORM=egl \
  python distill/prep_data.py --images distill/data/training/rgb \
                              --out distill/data/boxes.npz

# 3. distil (~125 ms/step at batch 16 on an RTX 5090 → 17 min/epoch)
CUDA_VISIBLE_DEVICES=0 PYGLET_HEADLESS=true PYOPENGL_PLATFORM=egl \
  python distill/train.py --boxes distill/data/boxes.npz \
                          --depth 16 --batch 16 --steps 40000 \
                          --out distill/runs/d16

# 4. score against the teacher
CUDA_VISIBLE_DEVICES=0 PYGLET_HEADLESS=true PYOPENGL_PLATFORM=egl \
  python distill/evaluate.py --ckpt distill/runs/d16/student_040000.pt
```

Boxes are cached rather than pixels so the loader can re-crop with jitter each epoch;
the crop reproduces `ViTDetDataset` exactly (rescale_factor 2.0, warp to 256×256,
normalise, hand back the 256×192 slice the ViT consumes), so the student trains on
the framing it will meet at inference.

## Evaluation

`evaluate.py` reports the quantities the fusion actually consumes — 2D pixels, bone
lengths, wrist-relative 3D — and flags absolute wrist error as UNUSED. It also
evaluates the teacher against itself for a noise floor, because inference is not
bit-reproducible on these GPUs.

## Gotchas found the hard way

- **`ViT.train()` (`wilor/models/backbones/vit.py:407`) does not return `self`**, so
  `.eval()` returns `None`. Never chain `model.half().to(dev).eval()` — you get
  `None` silently.
- The ViT's forward is re-implemented in `student.py::hidden_states` to expose
  intermediate states (upstream offers no hook). It was verified equivalent: "mine vs
  upstream" differs by 9.998e-02 on `img_feat` while "upstream vs itself" differs by
  9.975e-02 — i.e. the gap is entirely GPU non-determinism.
- A dataset smaller than the batch yields nothing with `drop_last=True` and the outer
  `while` loop spins forever; `train.py` now raises instead.

## Expected payoff

Depth halved ≈ 2× on the backbone. Stacked with the TensorRT FP8 engine (2.02× over
torch FP16, measured separately) the backbone should land near ~2.6 ms at batch 2,
versus 10.4 ms for the torch FP16 teacher.
