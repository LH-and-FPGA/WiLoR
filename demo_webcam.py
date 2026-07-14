import os
import sys
# Must be set before torch is imported: lets unsupported MPS ops fall back to CPU
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import argparse
import time
import cv2
import numpy as np
import torch
from torch.utils.data import default_collate

from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import Renderer, cam_crop_to_full
from ultralytics import YOLO

LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)

# 21-joint hand skeleton: wrist + 4 joints per finger
HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4),
              (0, 5), (5, 6), (6, 7), (7, 8),
              (0, 9), (9, 10), (10, 11), (11, 12),
              (0, 13), (13, 14), (14, 15), (15, 16),
              (0, 17), (17, 18), (18, 19), (19, 20)]
FINGER_COLORS = [(0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 0), (255, 128, 0)]


def project_to_image(points, cam_t, focal_length, img_w, img_h):
    points = points + cam_t
    x = focal_length * points[:, 0] / points[:, 2] + img_w / 2.
    y = focal_length * points[:, 1] / points[:, 2] + img_h / 2.
    return np.stack([x, y], axis=1)


def draw_hand_skeleton(frame, kpts_2d, is_right):
    for i, (a, b) in enumerate(HAND_BONES):
        color = FINGER_COLORS[i // 4]
        pa = tuple(np.round(kpts_2d[a]).astype(int))
        pb = tuple(np.round(kpts_2d[b]).astype(int))
        cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)
    for p in kpts_2d:
        cv2.circle(frame, tuple(np.round(p).astype(int)), 3, (255, 255, 255), -1, cv2.LINE_AA)
    wrist = tuple(np.round(kpts_2d[0]).astype(int))
    cv2.putText(frame, 'R' if is_right else 'L', (wrist[0] + 8, wrist[1] + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def process_frame(frame, model, model_cfg, detector, device, rescale_factor, conf,
                  renderer=None):
    """Run detection + WiLoR on one BGR frame, return annotated BGR frame."""
    detections = detector(frame, conf=conf, verbose=False)[0]
    bboxes, is_right = [], []
    for det in detections:
        box = det.boxes.data.cpu().detach().squeeze().numpy()
        is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
        bboxes.append(box[:4].tolist())
    if len(bboxes) == 0:
        return frame

    boxes = np.stack(bboxes)
    right = np.stack(is_right)
    dataset = ViTDetDataset(model_cfg, frame, boxes, right, rescale_factor=rescale_factor)
    batch = default_collate([dataset[i] for i in range(len(dataset))])
    # MPS does not support float64 tensors
    batch = {k: v.float() if torch.is_tensor(v) and v.dtype == torch.float64 else v
             for k, v in batch.items()}
    batch = recursive_to(batch, device)

    with torch.no_grad():
        out = model(batch)

    multiplier = (2 * batch['right'] - 1)
    pred_cam = out['pred_cam'].float()
    pred_cam[:, 1] = multiplier * pred_cam[:, 1]
    box_center = batch['box_center'].float()
    box_size = batch['box_size'].float()
    img_size = batch['img_size'].float()
    img_w, img_h = frame.shape[1], frame.shape[0]
    scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * max(img_w, img_h)
    pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size,
                                       scaled_focal_length).detach().cpu().numpy()

    all_verts, all_cam_t, all_right = [], [], []
    for n in range(len(dataset)):
        hand_right = batch['right'][n].cpu().numpy()
        cam_t = pred_cam_t_full[n]
        joints = out['pred_keypoints_3d'][n].detach().float().cpu().numpy()
        joints[:, 0] = (2 * hand_right - 1) * joints[:, 0]

        if renderer is not None:
            verts = out['pred_vertices'][n].detach().float().cpu().numpy()
            verts[:, 0] = (2 * hand_right - 1) * verts[:, 0]
            all_verts.append(verts)
            all_cam_t.append(cam_t)
            all_right.append(hand_right)
        else:
            kpts_2d = project_to_image(joints, cam_t, scaled_focal_length, img_w, img_h)
            draw_hand_skeleton(frame, kpts_2d, hand_right)

    if renderer is not None and len(all_verts) > 0:
        cam_view = renderer.render_rgba_multiple(
            all_verts, cam_t=all_cam_t, render_res=[img_w, img_h], is_right=all_right,
            mesh_base_color=LIGHT_PURPLE, scene_bg_color=(1, 1, 1),
            focal_length=scaled_focal_length)
        input_img = frame.astype(np.float32)[:, :, ::-1] / 255.0
        overlay = input_img * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
        frame = np.ascontiguousarray((255 * overlay[:, :, ::-1]).astype(np.uint8))

    return frame


def main():
    parser = argparse.ArgumentParser(description='WiLoR realtime webcam demo')
    parser.add_argument('--camera', type=int, default=0, help='Camera index for cv2.VideoCapture')
    parser.add_argument('--width', type=int, default=1280, help='Capture width')
    parser.add_argument('--height', type=int, default=720, help='Capture height')
    parser.add_argument('--conf', type=float, default=0.3, help='Detector confidence threshold')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'mps', 'cpu'])
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render full mesh overlay instead of skeleton (slower)')
    parser.add_argument('--mirror', action='store_true', default=False,
                        help='Mirror the camera image (selfie view)')
    parser.add_argument('--test_image', type=str, default=None,
                        help='Run a single image through the pipeline and save the result (no camera)')
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

    renderer = Renderer(model_cfg, faces=model.mano.faces) if args.render else None

    if args.test_image is not None:
        frame = cv2.imread(args.test_image)
        if frame is None:
            sys.exit(f'Could not read {args.test_image}')
        t0 = time.time()
        result = process_frame(frame, model, model_cfg, detector, device,
                               args.rescale_factor, args.conf, renderer)
        print(f'Frame processed in {time.time() - t0:.2f}s')
        out_path = 'webcam_test_out.jpg'
        cv2.imwrite(out_path, result)
        print(f'Saved {out_path}')
        return

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit('Could not open camera. Check that Terminal has camera permission '
                 '(System Settings -> Privacy & Security -> Camera).')

    print('Press q or ESC to quit.')
    fps = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.mirror:
            frame = cv2.flip(frame, 1)

        t0 = time.time()
        frame = process_frame(frame, model, model_cfg, detector, device,
                              args.rescale_factor, args.conf, renderer)
        dt = time.time() - t0
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt
        cv2.putText(frame, f'{fps:.1f} FPS', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('WiLoR webcam', frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
