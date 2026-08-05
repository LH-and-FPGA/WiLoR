"""Distil WiLoR's 32-layer ViT-H backbone into a 16-layer student.

Nothing outside `model.backbone` changes: RefineNet and both MANO passes stay frozen
and shared, so a student whose ViT outputs match the teacher's reproduces the whole
pipeline.  That is also why this needs no ground-truth labels -- the teacher supplies
every target, so any collection of hand images will do.

Run:
  CUDA_VISIBLE_DEVICES=0 PYGLET_HEADLESS=true PYOPENGL_PLATFORM=egl \
    python distill/train.py --boxes distill/data/boxes.npz --steps 20000

Evaluation during training reuses the paired-deviation harness from the analysis
work, reporting the quantities isoswarm's fusion actually consumes (2D pixels, bone
lengths, wrist-relative 3D) rather than absolute depth.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

WILOR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WILOR_DIR))
sys.path.insert(0, str(WILOR_DIR / "distill"))


# --------------------------------------------------------------------------- #
class CropDataset:
    """Crops built the way inference builds them, with light jitter.

    Boxes come from prep_data.py (the shipped detector, run once).  The crop
    reproduces ViTDetDataset: pad the box by rescale_factor, warp to 256x256,
    normalise, then hand back the 256x192 slice the ViT actually consumes.
    """

    def __init__(self, boxes_npz, img_size=256, rescale=2.0, jitter=True):
        import cv2
        from wilor.datasets.utils import expand_to_aspect_ratio
        self.cv2 = cv2
        self._expand = expand_to_aspect_ratio
        d = np.load(boxes_npz, allow_pickle=True)
        self.paths = d["paths"]
        self.boxes = d["boxes"]
        self.rights = d["rights"]
        self.img_size = img_size
        self.rescale = rescale
        self.jitter = jitter
        self.mean = 255.0 * np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = 255.0 * np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from wilor.datasets.utils import generate_image_patch_cv2
        cv2 = self.cv2
        img = cv2.imread(str(self.paths[idx]))
        if img is None:
            img = np.zeros((224, 224, 3), np.uint8)
        box = self.boxes[idx].astype(np.float64)
        centre = (box[2:4] + box[0:2]) / 2.0
        scale = self.rescale * (box[2:4] - box[0:2]) / 200.0
        rng = np.random
        if self.jitter:
            scale = scale * rng.uniform(0.9, 1.1)
            centre = centre + rng.uniform(-0.04, 0.04, 2) * (box[2:4] - box[0:2])
        bbox = self._expand(scale * 200, target_aspect_ratio=[192, 256]).max()
        flip = self.rights[idx] == 0
        patch, _ = generate_image_patch_cv2(img, centre[0], centre[1], bbox, bbox,
                                            self.img_size, self.img_size, flip, 1.0, 0,
                                            border_mode=cv2.BORDER_CONSTANT)
        patch = patch[:, :, ::-1].astype(np.float32)          # BGR -> RGB
        if self.jitter:
            patch *= rng.uniform(0.85, 1.15)                   # mild photometric jitter
        patch = (patch - self.mean) / self.std
        patch = np.transpose(patch, (2, 0, 1))
        return np.ascontiguousarray(patch[:, :, 32:-32])       # 3 x 256 x 192


def collate(batch):
    import torch
    return torch.from_numpy(np.stack(batch))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", default=str(WILOR_DIR / "distill/data/boxes.npz"))
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--out", default=str(WILOR_DIR / "distill/runs/d16"))
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    os.chdir(WILOR_DIR)
    os.environ.setdefault("PYGLET_HEADLESS", "true")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    import torch
    from torch.utils.data import DataLoader
    from wilor.models import load_wilor

    from student import DistillLoss, hidden_states, joints_from_backbone_out, make_student

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, cfg = load_wilor(checkpoint_path="./pretrained_models/wilor_final.ckpt",
                            cfg_path="./pretrained_models/model_config.yaml")
    # NB: ViT.train() (vit.py:407) does not return self, so .eval() returns None.
    # Never chain these calls.
    model.to(device)
    model.eval()
    teacher = model.backbone
    for p in model.parameters():
        p.requires_grad_(False)

    student, keep = make_student(teacher, depth=args.depth)
    student.to(device)
    student.train()
    for p in student.parameters():
        p.requires_grad_(True)
    print(f"student depth {args.depth} initialised from teacher layers {keep}")
    print(f"  teacher {sum(p.numel() for p in teacher.parameters())/1e6:.1f} M -> "
          f"student {sum(p.numel() for p in student.parameters())/1e6:.1f} M")

    ds = CropDataset(args.boxes)
    print(f"{len(ds)} crops")
    if len(ds) < args.batch:
        raise SystemExit(f"dataset has {len(ds)} crops but batch is {args.batch}; "
                         "with drop_last=True the loader yields nothing and the "
                         "training loop would spin forever")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    collate_fn=collate, drop_last=True, persistent_workers=args.workers > 0,
                    pin_memory=True)

    crit = DistillLoss()
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup)
        * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))

    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        student.load_state_dict(ck["student"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
        print(f"resumed from {args.resume} at step {start}")

    focal = cfg.EXTRA.FOCAL_LENGTH * torch.ones(args.batch, 2, device=device)
    t_layers = keep                    # student block k <-> teacher block keep[k]
    log_path = out_dir / "log.jsonl"
    step = start
    t0 = time.perf_counter()
    running = {}

    while step < args.steps:
        for x in dl:
            if step >= args.steps:
                break
            x = x.to(device, non_blocking=True)

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                t_feats, t_cam, t_img, t_states = hidden_states(teacher, x, t_layers)
                t_joints, _, t_kp2d = joints_from_backbone_out(
                    model, t_feats, t_cam, t_img.float(), focal[:x.shape[0]])

            with torch.autocast("cuda", dtype=torch.bfloat16):
                s_feats, s_cam, s_img, s_states = hidden_states(student, x, None)
                s_joints, _, s_kp2d = joints_from_backbone_out(
                    model, s_feats, s_cam, s_img.float(), focal[:x.shape[0]])
                loss, parts = crit(
                    {"img_feat": s_img, "feats": s_feats, "states": s_states,
                     "joints": s_joints, "kp2d": s_kp2d},
                    {"img_feat": t_img, "feats": t_feats, "states": t_states,
                     "joints": t_joints, "kp2d": t_kp2d})

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1

            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v
            running["total"] = running.get("total", 0.0) + float(loss)

            if step % 50 == 0:
                n = 50
                rec = {"step": step, "lr": sched.get_last_lr()[0],
                       "grad_norm": float(gn),
                       "sec_per_step": (time.perf_counter() - t0) / n,
                       **{k: v / n for k, v in running.items()}}
                with open(log_path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(f"step {step:6d}  loss {rec['total']:7.3f}  feat {rec['feat']:6.4f}  "
                      f"kp_rel {rec['kp_rel']:6.2f} mm  kp2d {rec['kp2d_px']:6.2f} px  "
                      f"{rec['sec_per_step']*1000:5.0f} ms/step", flush=True)
                running = {}
                t0 = time.perf_counter()

            if step % args.save_every == 0 or step == args.steps:
                ckpt = out_dir / f"student_{step:06d}.pt"
                torch.save({"student": student.state_dict(), "opt": opt.state_dict(),
                            "step": step, "depth": args.depth, "keep": keep}, ckpt)
                print(f"  saved {ckpt}")

    print("done")


if __name__ == "__main__":
    main()
