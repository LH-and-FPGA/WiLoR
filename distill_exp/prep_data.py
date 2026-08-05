import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

WILOR_DIR = Path(__file__).resolve().parent.parent
print(WILOR_DIR)
DATA_DIR = WILOR_DIR / "distill" / "data"
PRETRAINED_MODELS_DIR = WILOR_DIR / "pretrained_models"
IMG_DIR = DATA_DIR / "training" / "rgb"
OUT_PATH = WILOR_DIR / "distill_exp" / "data" / "boxes.npz"
batch = 64


def main():
    detector = YOLO(PRETRAINED_MODELS_DIR / "detector.pt").to("cuda")
    paths = sorted(Path(IMG_DIR).glob("*.jpg")) + sorted(Path(IMG_DIR).glob("*.png"))
    print(f"{len(paths)} images in {IMG_DIR}")
    keep_paths, boxes, rights = [], [], []

    for i in range(0, len(paths), batch):
        chunk = [str(p) for p in paths[i : i + batch]]
        results = detector(chunk, verbose=False)

        for p, r in zip(chunk, results):
            if len(r.boxes) == 0:
                continue
            b = r.boxes.data.cpu().numpy()
            j = int(b[:, 4].argmax())
            # as we have b of size (N, 6)
            # [x1, y1, x2, y2, conf, cls]
            # get the conf col and find the largest
            keep_paths.append(p)
            boxes.append(b[j, :4])
            rights.append(float(r.boxes.cls[j]))
        if (i // 64) % 20 == 0:
            print(f"  {i+len(chunk)}/{len(paths)}  kept {len(keep_paths)}", flush=True)
    boxes = np.stack(boxes).astype(np.float32)
    rights = np.array(rights, dtype=np.float32)

    # 存相对路径，npz 才能跨机器用；训练时再拼回去
    rel_paths = [str(Path(p).relative_to(WILOR_DIR)) for p in keep_paths]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_PATH, paths=np.array(rel_paths), boxes=boxes, rights=rights)
    print(
        f"kept {len(keep_paths)}/{len(paths)} images ({100*len(keep_paths)/len(paths):.1f}%)"
    )
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
