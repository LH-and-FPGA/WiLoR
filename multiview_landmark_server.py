"""WiLoR multi-camera UDP v2 sender for calibrated isoswarm fusion.

Run this inside the WiLoR environment from the WiLoR repository root. It owns
all USB cameras, grabs one near-synchronous set each cycle, runs the existing
WiLoR pipeline once per view using a single loaded model, and emits exactly one
v2 JSON bundle. The matching isoswarm backend performs calibrated association,
triangulation, quality gates and temporal filtering.

Example:
    python multiview_landmark_server.py \
        --camera cam_left=/dev/v4l/by-id/usb-left \
        --camera cam_right=/dev/v4l/by-id/usb-right \
        --port 9402 --device cuda --fast
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from collections.abc import Sequence

# Must be set before torch is imported, matching landmark_server.py.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import torch
from ultralytics import YOLO
from wilor.models import load_wilor

from demo_webcam import draw_hand_skeleton
from landmark_server import extract_hands


def parse_camera_spec(value: str) -> tuple[str, int | str]:
    """Parse ``camera_id=device`` without changing stable device paths."""

    if "=" not in value:
        raise argparse.ArgumentTypeError("camera must have the form camera_id=device")
    camera_id, device = (part.strip() for part in value.split("=", maxsplit=1))
    if not camera_id or not device:
        raise argparse.ArgumentTypeError("camera_id and device must both be non-empty")
    if device.lstrip("-").isdigit():
        return camera_id, int(device)
    return camera_id, device


def open_cameras(
    specs: Sequence[tuple[str, int | str]], width: int, height: int, fps: int
) -> list[tuple[str, cv2.VideoCapture]]:
    """Open all cameras or close every already-open camera before raising."""

    cameras: list[tuple[str, cv2.VideoCapture]] = []
    try:
        for camera_id, device in specs:
            capture = cv2.VideoCapture(device)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            capture.set(cv2.CAP_PROP_FPS, fps)
            if not capture.isOpened():
                raise RuntimeError(f"could not open {camera_id!r} at {device!r}")
            cameras.append((camera_id, capture))
        return cameras
    except Exception:
        for _, capture in cameras:
            capture.release()
        raise


def grab_synchronized(
    cameras: Sequence[tuple[str, cv2.VideoCapture]],
) -> tuple[list[tuple[str, float, object]], float] | None:
    """Grab every view before retrieval to minimize software capture skew."""

    timestamps: list[tuple[str, float]] = []
    for camera_id, capture in cameras:
        if not capture.grab():
            return None
        timestamps.append((camera_id, time.monotonic()))

    frames: list[tuple[str, float, object]] = []
    for (camera_id, capture), (_, timestamp) in zip(cameras, timestamps):
        ok, frame = capture.retrieve()
        if not ok:
            return None
        frames.append((camera_id, timestamp, frame))
    bundle_timestamp = sum(timestamp for _, timestamp in timestamps) / len(timestamps)
    return frames, bundle_timestamp


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera",
        action="append",
        type=parse_camera_spec,
        required=True,
        metavar="ID=DEVICE",
        help="repeat for every view; device may be an index or /dev/v4l/by-id path",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9402)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--fast", action="store_true", help="use FP16 on CUDA/MPS")
    parser.add_argument("--faster", action="store_true", help="FP16 plus ViT layer dropping")
    args = parser.parse_args()

    ids = [camera_id for camera_id, _ in args.camera]
    if len(ids) != len(set(ids)):
        parser.error("camera ids must be unique")
    if len(args.camera) < 2:
        parser.error("at least two --camera entries are required for multi-view fusion")

    device = choose_device(args.device)
    print(f"Using device: {device}")
    model, model_cfg = load_wilor(
        checkpoint_path="./pretrained_models/wilor_final.ckpt",
        cfg_path="./pretrained_models/model_config.yaml",
    )
    detector = YOLO("./pretrained_models/detector.pt")
    model = model.to(device)
    detector = detector.to(device)
    model.eval()

    use_fp16 = args.fast or args.faster
    if use_fp16 and device.type == "cpu":
        print("--fast/--faster need CUDA or MPS; running FP32 on CPU instead")
        use_fp16 = False
    if use_fp16:
        model = model.half()
        if args.faster:
            model.backbone.skip_blocks = True
        print(f"Inference mode: FP16{' + layer dropping' if args.faster else ''}")
    else:
        print("Inference mode: FP32")

    try:
        cameras = open_cameras(args.camera, args.width, args.height, args.fps)
    except RuntimeError as error:
        sys.exit(str(error))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.host, args.port)
    print(f"Streaming WiLoR v2 bundles to udp://{args.host}:{args.port}")

    sequence = 0
    sent = 0
    last_report = time.monotonic()
    try:
        while True:
            grabbed = grab_synchronized(cameras)
            if grabbed is None:
                print("a camera failed to grab/retrieve; stopping", file=sys.stderr)
                break
            frames, bundle_timestamp = grabbed
            views = []
            for camera_id, timestamp, frame in frames:
                hands = extract_hands(
                    frame,
                    model,
                    model_cfg,
                    detector,
                    device,
                    args.rescale_factor,
                    args.conf,
                    fp16=use_fp16,
                )
                views.append(
                    {
                        "camera_id": camera_id,
                        "t": timestamp,
                        "hands": [
                            {key: hand[key] for key in ("lm", "world", "handedness", "conf")}
                            for hand in hands
                        ],
                    }
                )
                if args.preview:
                    preview = cv2.flip(frame, 1)
                    for hand in hands:
                        points = hand["kpts_2d"].copy()
                        points[:, 0] = frame.shape[1] - points[:, 0]
                        draw_hand_skeleton(preview, points, hand["right"])
                    cv2.imshow(f"WiLoR {camera_id}", preview)

            packet = {"version": 2, "seq": sequence, "t": bundle_timestamp, "views": views}
            sock.sendto(json.dumps(packet).encode("utf-8"), target)
            sequence += 1
            sent += 1

            now = time.monotonic()
            if now - last_report >= 5.0:
                view_counts = ", ".join(
                    f"{view['camera_id']}={len(view['hands'])}" for view in views
                )
                print(f"{sent / (now - last_report):.1f} bundle/s; hands: {view_counts}")
                sent = 0
                last_report = now
            if args.preview and (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        for _, capture in cameras:
            capture.release()
        sock.close()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

