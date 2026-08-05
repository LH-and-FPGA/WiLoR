"""Score a distilled student against the teacher it was distilled from.

Self-contained on purpose: the repo should not depend on the throwaway analysis
scripts this work grew out of.

Metrics are the ones the downstream multi-view fusion actually consumes:
  px        2D landmark deviation, pixels        -- what triangulation eats
  bone_mm   bone-length deviation, millimetres    -- what the bone prior eats
  mm_rel    wrist-relative 3D, millimetres        -- articulation
Absolute depth / global translation is reported but flagged UNUSED: the fusion code
touches the per-view monocular 3D only through bone lengths and a scale-invariant
Kabsch fit, so a large wrist error there means much less than it looks.

Because inference is not bit-reproducible on these GPUs, the teacher is also
evaluated against itself to give a noise floor.

Run:
  CUDA_VISIBLE_DEVICES=0 PYGLET_HEADLESS=true PYOPENGL_PLATFORM=egl \
    python distill/evaluate.py --ckpt distill/runs/d16/student_020000.pt \
                               --images demo_img
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

WILOR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WILOR_DIR))
sys.path.insert(0, str(WILOR_DIR / "distill"))

HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
              (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def bone_lengths(p):
    return np.array([np.linalg.norm(p[a] - p[b]) for a, b in HAND_BONES])


def infer_frame(frame, model, cfg, detector, device, torch):
    """One frame through detector + model, returning per-hand 2D px and camera-frame 3D."""
    from torch.utils.data._utils.collate import default_collate
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils import recursive_to
    from wilor.utils.renderer import cam_crop_to_full

    dets = detector(frame, conf=0.3, verbose=False)[0]
    boxes, right, conf = [], [], []
    for d in dets:
        b = d.boxes.data.cpu().numpy().squeeze()
        boxes.append(b[:4].tolist())
        right.append(d.boxes.cls.cpu().numpy().squeeze().item())
        conf.append(float(d.boxes.conf.cpu().numpy().squeeze().item()))
    if not boxes:
        return []
    ds = ViTDetDataset(cfg, frame, np.stack(boxes), np.stack(right), rescale_factor=2.0)
    batch = default_collate([ds[i] for i in range(len(ds))])
    batch = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
             for k, v in batch.items()}
    batch = recursive_to(batch, device)
    with torch.no_grad():
        out = model(batch)
    mult = 2 * batch["right"] - 1
    pred_cam = out["pred_cam"].float()
    pred_cam[:, 1] = mult * pred_cam[:, 1]
    w, h = frame.shape[1], frame.shape[0]
    sfl = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * max(w, h)
    cam_t = cam_crop_to_full(pred_cam, batch["box_center"].float(),
                             batch["box_size"].float(), batch["img_size"].float(),
                             sfl).detach().cpu().numpy()
    hands = []
    for n in range(len(ds)):
        hr = float(batch["right"][n].cpu().numpy())
        j = out["pred_keypoints_3d"][n].detach().float().cpu().numpy()
        j[:, 0] = (2 * hr - 1) * j[:, 0]
        world = j + cam_t[n]
        z = np.where(np.abs(world[:, 2]) < 1e-9, 1e-9, world[:, 2])
        px = np.stack([sfl * world[:, 0] / z + w / 2.0,
                       sfl * world[:, 1] / z + h / 2.0], -1)
        hands.append({"px": px, "world_m": world, "conf": conf[n]})
    return hands


def compare(ref, test):
    """Paired deviation, hands matched by wrist proximity (never by handedness label:
    WiLoR is a right-hand MANO model and labels both hands 'Right')."""
    px, rel, wrist, bone = [], [], [], []
    for img, rhs in ref.items():
        ths, used = test.get(img, []), set()
        for rh in rhs:
            best, bd = None, None
            for j, th in enumerate(ths):
                if j in used:
                    continue
                d = float(np.linalg.norm(th["px"][0] - rh["px"][0]))
                if bd is None or d < bd:
                    best, bd = j, d
            if best is None:
                continue
            used.add(best)
            th = ths[best]
            px.append(np.linalg.norm(th["px"] - rh["px"], axis=-1))
            rr = (th["world_m"] - th["world_m"][0]) - (rh["world_m"] - rh["world_m"][0])
            rel.append(np.linalg.norm(rr, axis=-1) * 1000)
            wrist.append(np.linalg.norm(th["world_m"][0] - rh["world_m"][0]) * 1000)
            bone.append(np.abs(bone_lengths(th["world_m"])
                               - bone_lengths(rh["world_m"])) * 1000)
    if not px:
        return {"hands": 0}
    px, rel, bone = np.concatenate(px), np.concatenate(rel), np.concatenate(bone)
    return {"hands": len(rel) // 21,
            "px_mean": float(px.mean()), "px_p95": float(np.percentile(px, 95)),
            "px_max": float(px.max()),
            "bone_mm_mean": float(bone.mean()), "bone_mm_max": float(bone.max()),
            "mm_rel_mean": float(rel.mean()), "mm_rel_max": float(rel.max()),
            "UNUSED_wrist_mm_mean": float(np.mean(wrist))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images", default=str(WILOR_DIR / "demo_img"))
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.chdir(WILOR_DIR)
    os.environ.setdefault("PYGLET_HEADLESS", "true")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    import cv2
    import torch
    from ultralytics import YOLO
    from wilor.models import load_wilor

    from student import make_student

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    orig = torch.load

    def trusted(*a, **k):
        k.setdefault("weights_only", False)
        return orig(*a, **k)

    torch.load = trusted
    try:
        detector = YOLO("./pretrained_models/detector.pt").to(device)
    finally:
        torch.load = orig

    frames = {p.name: cv2.imread(str(p)) for p in sorted(Path(args.images).glob("*"))
              if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    print(f"{len(frames)} eval images")

    def latency(m, dtype):
        lat = {}
        with torch.no_grad():
            for b in [1, 2, 6]:
                x = torch.randn(b, 3, 256, 192, device=device, dtype=dtype)
                for _ in range(6):
                    m.backbone(x)
                torch.cuda.synchronize()
                s = []
                for _ in range(args.iters):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    m.backbone(x)
                    torch.cuda.synchronize()
                    s.append((time.perf_counter() - t) * 1000)
                lat[f"batch{b}_ms"] = statistics.median(s)
        return lat

    model, cfg = load_wilor(checkpoint_path="./pretrained_models/wilor_final.ckpt",
                            cfg_path="./pretrained_models/model_config.yaml")
    model.to(device)
    model.eval()                      # ViT.train() returns None -- never chain this

    teacher_out = {f: infer_frame(fr, model, cfg, detector, device, torch)
                   for f, fr in frames.items()}
    teacher_lat = latency(model, torch.float32)
    noise = compare(teacher_out, {f: infer_frame(fr, model, cfg, detector, device, torch)
                                  for f, fr in frames.items()})
    print(f"teacher fp32 {teacher_lat}")
    print(f"noise floor: px {noise['px_mean']:.3f}  bone {noise['bone_mm_mean']:.3f} mm  "
          f"rel {noise['mm_rel_mean']:.3f} mm")

    ck = torch.load(args.ckpt, map_location="cpu")
    student, keep = make_student(model.backbone, depth=ck["depth"])
    student.load_state_dict(ck["student"])
    student.to(device)
    student.eval()
    model.backbone = student
    student_out = {f: infer_frame(fr, model, cfg, detector, device, torch)
                   for f, fr in frames.items()}
    student_lat = latency(model, torch.float32)
    dev = compare(teacher_out, student_out)

    print(f"\nstudent depth {ck['depth']} @ step {ck['step']}")
    print(f"  latency {student_lat}  ({teacher_lat['batch2_ms']/student_lat['batch2_ms']:.2f}x "
          f"vs teacher at batch 2)")
    print(f"  px  mean {dev['px_mean']:7.3f}  p95 {dev['px_p95']:7.3f}  max {dev['px_max']:7.2f}")
    print(f"  bone      {dev['bone_mm_mean']:7.3f} mm  max {dev['bone_mm_max']:7.3f} mm")
    print(f"  mm_rel    {dev['mm_rel_mean']:7.3f} mm  max {dev['mm_rel_max']:7.2f} mm")
    print(f"  (UNUSED wrist {dev['UNUSED_wrist_mm_mean']:.1f} mm)")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"ckpt": args.ckpt, "step": ck["step"], "depth": ck["depth"],
             "teacher_ms": teacher_lat, "student_ms": student_lat,
             "noise_floor": noise, "deviation": dev}, indent=2, default=float))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
