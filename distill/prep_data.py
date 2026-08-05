"""Turn FreiHAND into crops that match WiLoR's inference-time input distribution.

Distillation only works if the student sees what it will see at deployment.  The
real pipeline detects a box with YOLO, pads it by rescale_factor=2.0, and warps to
256x256 before the ViT slices it to 256x192.  FreiHAND ships tight 224x224 hand
crops, so feeding those raw would train the student on a framing it never meets.

So: run the shipped detector over every image once, cache the boxes, and let the
training loader do the real crop (with light jitter) on the fly.  Caching boxes
rather than pixels keeps this to a few MB and leaves room for augmentation.

Images with no detection are dropped -- if the teacher cannot see a hand there is
nothing to distil from.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

WILOR_DIR = Path("/workspace/WiLoR")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="dir of FreiHAND rgb images")
    ap.add_argument("--out", default="/workspace/note/distill/data/boxes.npz")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    sys.path.insert(0, str(WILOR_DIR))
    os.chdir(WILOR_DIR)
    os.environ.setdefault("PYGLET_HEADLESS", "true")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    import torch
    from ultralytics import YOLO

    orig = torch.load

    def trusted(*a, **k):
        k.setdefault("weights_only", False)
        return orig(*a, **k)

    torch.load = trusted
    try:
        det = YOLO("./pretrained_models/detector.pt").to("cuda")
    finally:
        torch.load = orig

    paths = sorted(Path(args.images).glob("*.jpg")) + sorted(Path(args.images).glob("*.png"))
    if args.limit:
        paths = paths[:args.limit]
    print(f"{len(paths)} images in {args.images}")

    keep_paths, boxes, rights = [], [], []
    for i in range(0, len(paths), args.batch):
        chunk = [str(p) for p in paths[i:i + args.batch]]
        results = det(chunk, conf=args.conf, verbose=False)
        for p, r in zip(chunk, results):
            if len(r.boxes) == 0:
                continue
            b = r.boxes.data.cpu().numpy()
            j = int(b[:, 4].argmax())               # highest confidence
            keep_paths.append(p)
            boxes.append(b[j, :4])
            rights.append(float(r.boxes.cls.cpu().numpy()[j]))
        if (i // args.batch) % 20 == 0:
            print(f"  {i+len(chunk)}/{len(paths)}  kept {len(keep_paths)}", flush=True)

    boxes = np.stack(boxes).astype(np.float32)
    rights = np.array(rights, dtype=np.float32)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, paths=np.array(keep_paths), boxes=boxes, rights=rights)
    print(f"kept {len(keep_paths)}/{len(paths)} images ({100*len(keep_paths)/len(paths):.1f}%)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
