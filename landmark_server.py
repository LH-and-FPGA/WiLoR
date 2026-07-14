"""WiLoR hand-landmark UDP server for isoswarm.

Owns the webcam, runs YOLO hand detection + WiLoR 3D pose estimation (the
same pipeline as demo_webcam.py), and streams 21-point hand landmarks as one
JSON datagram per processed frame to isoswarm's WilorBridgeLandmarkDetector.

Run inside the `wilor` conda environment from the WiLoR repo root:

    conda run -n wilor python landmark_server.py [--preview]

Datagram schema (matches isoswarm/landmark/wilor_bridge.py):

    {"seq": int, "t": float monotonic capture time, "camera_id": str,
     "hands": [{"lm": [[x,y,z]*21],     # normalized image coords, z relative
                "world": [[x,y,z]*21],  # metric camera-frame (m)
                "handedness": "Right"|"Left", "conf": float}]}

WiLoR outputs MANO-ordered joints, which match the canonical MediaPipe
21-point order used by isoswarm — no reindexing is needed.
"""

import os
import sys
# Must be set before torch is imported: lets unsupported MPS ops fall back to CPU
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import argparse
import json
import socket
import time

import cv2
import numpy as np
import torch
from torch.utils.data import default_collate

from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset

from demo_webcam import draw_hand_skeleton, project_to_image
from ultralytics import YOLO


def extract_hands(frame, model, model_cfg, detector, device, rescale_factor, conf,
                  fp16=False):
    """Run detection + WiLoR on one BGR frame; return the per-hand payload list."""
    detections = detector(frame, conf=conf, verbose=False)[0]
    bboxes, is_right = [], []
    for det in detections:
        box = det.boxes.data.cpu().detach().squeeze().numpy()
        is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
        bboxes.append(box[:4].tolist())
    if len(bboxes) == 0:
        return []

    dataset = ViTDetDataset(model_cfg, frame, np.stack(bboxes), np.stack(is_right),
                            rescale_factor=rescale_factor, fp16=fp16)
    batch = default_collate([dataset[i] for i in range(len(dataset))])
    # Cast all float tensors to the model dtype (also avoids float64, which
    # MPS does not support).
    float_dtype = torch.float16 if fp16 else torch.float32
    batch = {k: v.to(float_dtype) if torch.is_tensor(v) and v.is_floating_point() else v
             for k, v in batch.items()}
    batch = recursive_to(batch, device)

    with torch.no_grad():
        out = model(batch)

    from wilor.utils.renderer import cam_crop_to_full
    multiplier = (2 * batch['right'] - 1)
    pred_cam = out['pred_cam'].float()
    pred_cam[:, 1] = multiplier * pred_cam[:, 1]
    img_w, img_h = frame.shape[1], frame.shape[0]
    scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * max(img_w, img_h)
    pred_cam_t_full = cam_crop_to_full(
        pred_cam, batch['box_center'].float(), batch['box_size'].float(),
        batch['img_size'].float(), scaled_focal_length).detach().cpu().numpy()

    hands = []
    for n in range(len(dataset)):
        hand_right = float(batch['right'][n].cpu().numpy())
        cam_t = pred_cam_t_full[n]
        joints = out['pred_keypoints_3d'][n].detach().float().cpu().numpy()
        joints[:, 0] = (2 * hand_right - 1) * joints[:, 0]

        kpts_2d = project_to_image(joints, cam_t, scaled_focal_length, img_w, img_h)

        # Normalized image coords; z = depth relative to the wrist (image-z proxy)
        lm = np.zeros((21, 3))
        lm[:, 0] = kpts_2d[:, 0] / img_w
        lm[:, 1] = kpts_2d[:, 1] / img_h
        lm[:, 2] = joints[:, 2] - joints[0, 2]

        world = joints + cam_t  # metric camera-frame position

        conf_score = 1.0  # WiLoR has no per-hand score; the detector conf gates entry
        hands.append({
            'lm': np.round(lm, 5).tolist(),
            'world': np.round(world, 5).tolist(),
            'handedness': 'Right' if hand_right > 0.5 else 'Left',
            'conf': conf_score,
            'kpts_2d': kpts_2d,  # stripped before sending; used by --preview
            'right': hand_right,
        })
    return hands


def main():
    parser = argparse.ArgumentParser(description='WiLoR landmark UDP server for isoswarm')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--conf', type=float, default=0.3)
    parser.add_argument('--rescale_factor', type=float, default=2.0)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'mps', 'cpu'])
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9401)
    parser.add_argument('--camera_id', type=str, default='wilor0')
    parser.add_argument('--preview', action='store_true', default=False,
                        help='Show a local window with the detected skeletons')
    parser.add_argument('--fast', action='store_true', default=False,
                        help='FP16 inference (MPS/CUDA): ~2.6x faster model forward, '
                             'negligible accuracy loss (~0.1mm)')
    parser.add_argument('--faster', action='store_true', default=False,
                        help='FP16 + ViT layer dropping: fastest, small accuracy cost '
                             '(~5mm max joint error); implies --fast')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('mps') if torch.backends.mps.is_available() else \
                 torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f'Using device: {device}')

    model, model_cfg = load_wilor(checkpoint_path='./pretrained_models/wilor_final.ckpt',
                                  cfg_path='./pretrained_models/model_config.yaml')
    detector = YOLO('./pretrained_models/detector.pt')
    model = model.to(device)
    detector = detector.to(device)
    model.eval()

    use_fp16 = args.fast or args.faster
    if use_fp16 and device.type == 'cpu':
        print('--fast/--faster need mps or cuda; running fp32 on cpu instead')
        use_fp16 = False
    if use_fp16:
        model = model.half()
        if args.faster:
            model.backbone.skip_blocks = True
        print(f'Inference mode: fp16{" + layer dropping" if args.faster else ""}')
    else:
        print('Inference mode: fp32')

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.host, args.port)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit('Could not open camera. Check that Terminal has camera permission '
                 '(System Settings -> Privacy & Security -> Camera).')

    print(f'Streaming landmarks to udp://{args.host}:{args.port} (Ctrl-C to stop)')
    seq = 0
    sent = 0
    last_report = time.monotonic()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t_capture = time.monotonic()

            hands = extract_hands(frame, model, model_cfg, detector, device,
                                  args.rescale_factor, args.conf, fp16=use_fp16)

            packet = {
                'seq': seq,
                't': t_capture,
                'camera_id': args.camera_id,
                'hands': [{k: h[k] for k in ('lm', 'world', 'handedness', 'conf')}
                          for h in hands],
            }
            sock.sendto(json.dumps(packet).encode('utf-8'), target)
            seq += 1
            sent += 1

            now = time.monotonic()
            if now - last_report >= 5.0:
                print(f'{sent / (now - last_report):.1f} packets/s, '
                      f'{len(hands)} hand(s) in last frame')
                sent = 0
                last_report = now

            if args.preview:
                # Mirrored view (like a mirror) — display only; the packets
                # keep the un-mirrored camera coordinates.
                view = cv2.flip(frame, 1)
                for h in hands:
                    kpts = h['kpts_2d'].copy()
                    kpts[:, 0] = frame.shape[1] - kpts[:, 0]
                    draw_hand_skeleton(view, kpts, h['right'])
                cv2.imshow('WiLoR landmark server', view)
                if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if args.preview:
            cv2.destroyAllWindows()
        sock.close()


if __name__ == '__main__':
    main()
