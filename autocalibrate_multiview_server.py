"""Calibrate a two-camera hand rig by waving a hand, then stream v2 bundles.

No printed target.  When both cameras see the same hand at the same instant,
WiLoR's 21 landmarks are a set of known-correspondence point pairs -- better
conditioned than a chessboard in one respect, because the correspondence is
given by MANO joint order rather than searched for.  Accumulating those pairs
while the hand moves through the shared view yields enough geometry to solve
the essential matrix, and from it the rotation and the *direction* of the
translation between the cameras.

Epipolar geometry cannot recover the length of that translation: a small rig
watching a small hand and a large rig watching a large hand produce identical
images.  That one number has to come from outside, so pass the measured
distance between the two lens centres via --baseline.  Measure it with a
ruler; 1 mm on a 180 mm baseline is a 0.5% scale error, while deriving scale
from WiLoR's monocular depth would be off by considerably more.  Scale matters
downstream: isoswarm gates fused metric positions against a fixed working
volume and silently drops commands that fall outside it.

Intrinsics are not solved.  Both cameras use the nominal pinhole model WiLoR
itself assumes when it projects joints to the image, so the calibration stays
consistent with the landmarks feeding it.

After writing the calibration the process keeps the cameras and the loaded
model and continues straight into streaming, so the rig is never disturbed
between calibrating and running.

    conda run -n wilor python autocalibrate_multiview_server.py \
        --camera cam_left=/dev/video0 \
        --camera cam_right=/dev/video2 \
        --baseline 0.185 \
        --out ../isoswarm/configs/calibration/hand_rig.yaml \
        --device cuda
"""

from __future__ import annotations

import argparse
import os
import socket
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import yaml

from landmark_server import extract_hands
from multiview_landmark_server import (
    choose_device,
    grab_synchronized,
    load_pipeline,
    open_cameras,
    parse_camera_spec,
    stream_bundles,
)

# Coverage is tracked on a coarse grid because essential-matrix estimation is
# ill-conditioned when correspondences sit in one part of the image.
_COVERAGE_GRID = (4, 3)
# RANSAC threshold in pixels for the essential matrix.  WiLoR landmarks are
# noisier than chessboard corners, so this is looser than a board would need.
_RANSAC_THRESHOLD_PX = 2.0
_RANSAC_CONFIDENCE = 0.999


def nominal_intrinsics(model_cfg, width: int, height: int) -> np.ndarray:
    """Build the pinhole matrix WiLoR itself uses to project joints.

    Reusing WiLoR's own convention keeps the calibration consistent with the
    landmarks that will be triangulated against it, which matters more here
    than matching the physical lens exactly.
    """

    focal = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * max(width, height)
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def check_cameras_see_something(cameras) -> None:
    """Fail fast on a lens that is covered or a driver returning blank frames.

    A camera behind a closed privacy shutter returns a perfectly uniform frame.
    It detects no hands, so calibration would simply never accumulate samples
    and the operator would be left waving at nothing.
    """

    for camera_id, capture in cameras:
        for _ in range(10):
            capture.grab()
        ok, frame = capture.retrieve()
        if not ok:
            sys.exit(f"{camera_id}: could not read a frame")
        spread = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).std())
        if spread < 1.0:
            sys.exit(
                f"{camera_id}: frames are a uniform image (pixel std {spread:.2f}). "
                "The lens is covered or the driver is returning blank buffers -- "
                "check for a privacy shutter before calibrating."
            )


def single_hand(hands: list[dict]) -> dict | None:
    """Return the only hand in a view, or None when the pairing is ambiguous."""

    return hands[0] if len(hands) == 1 else None


def coverage_cells(points: np.ndarray, width: int, height: int) -> set[tuple[int, int]]:
    columns, rows = _COVERAGE_GRID
    xs = np.clip((points[:, 0] / width * columns).astype(int), 0, columns - 1)
    ys = np.clip((points[:, 1] / height * rows).astype(int), 0, rows - 1)
    return set(zip(xs.tolist(), ys.tolist()))


def collect_correspondences(
    cameras, infer, width: int, height: int, target_frames: int, preview: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate landmark pairs seen simultaneously by both cameras."""

    columns, rows = _COVERAGE_GRID
    total_cells = columns * rows
    left_points: list[np.ndarray] = []
    right_points: list[np.ndarray] = []
    covered: set[tuple[int, int]] = set()
    frames_used = 0
    rejected_multi = 0
    rejected_missing = 0

    print(
        f"\nWave one hand slowly through the shared field of view of both cameras.\n"
        f"Cover the whole frame -- corners as well as centre -- and vary distance.\n"
        f"Need {target_frames} paired frames and all {total_cells} coverage cells.\n"
        "Keep only ONE hand visible: two hands make the pairing ambiguous.\n"
        "Press q in the preview window to abort.\n"
    )

    while frames_used < target_frames or len(covered) < total_cells:
        grabbed = grab_synchronized(cameras)
        if grabbed is None:
            sys.exit("a camera failed to grab/retrieve during calibration")
        frames, _ = grabbed

        detections = []
        for camera_id, _, frame in frames:
            hands = infer(frame)
            detections.append((camera_id, frame, hands))

        chosen = [single_hand(hands) for _, _, hands in detections]
        if any(hand is None for hand in chosen):
            if any(len(hands) > 1 for _, _, hands in detections):
                rejected_multi += 1
            else:
                rejected_missing += 1
        elif chosen[0]["handedness"] != chosen[1]["handedness"]:
            # Different handedness means the two views disagree about what they
            # are looking at; pairing those landmarks would corrupt the solve.
            rejected_multi += 1
        else:
            left = np.asarray(chosen[0]["kpts_2d"], dtype=np.float64)
            right = np.asarray(chosen[1]["kpts_2d"], dtype=np.float64)
            left_points.append(left)
            right_points.append(right)
            covered |= coverage_cells(left, width, height)
            frames_used += 1

        if preview:
            panels = []
            for (camera_id, frame, hands), hand in zip(detections, chosen):
                panel = cv2.resize(frame, (640, 360))
                colour = (0, 200, 0) if hand is not None else (0, 0, 220)
                cv2.putText(panel, camera_id, (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, colour, 2)
                panels.append(panel)
            canvas = np.hstack(panels)
            cv2.putText(
                canvas,
                f"pairs {frames_used}/{target_frames}  cells {len(covered)}/{total_cells}",
                (12, 348), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            cv2.imshow("hand rig autocalibration", canvas)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

        if frames_used and frames_used % 25 == 0 and not preview:
            print(f"  pairs {frames_used}/{target_frames}, "
                  f"coverage {len(covered)}/{total_cells} cells")

    if preview:
        cv2.destroyAllWindows()
    if frames_used < 8:
        sys.exit(
            f"only {frames_used} usable frame pairs "
            f"({rejected_missing} with a hand missing from a view, "
            f"{rejected_multi} ambiguous) -- not enough to calibrate"
        )
    if len(covered) < total_cells:
        print(f"warning: only {len(covered)}/{total_cells} image regions covered; "
              "the solution will be poorly conditioned toward the uncovered edges",
              file=sys.stderr)

    print(f"collected {frames_used} pairs ({frames_used * 21} point correspondences), "
          f"coverage {len(covered)}/{total_cells}")
    return np.vstack(left_points), np.vstack(right_points)


def solve_extrinsics(
    left: np.ndarray, right: np.ndarray, k_matrix: np.ndarray, baseline_m: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Recover rotation and metric translation from landmark correspondences."""

    essential, mask = cv2.findEssentialMat(
        left, right, k_matrix, method=cv2.RANSAC,
        prob=_RANSAC_CONFIDENCE, threshold=_RANSAC_THRESHOLD_PX)
    if essential is None or essential.shape != (3, 3):
        sys.exit("essential matrix estimation failed; recapture with more varied motion")

    inlier_ratio = float(mask.sum()) / len(mask)
    print(f"essential matrix: {int(mask.sum())}/{len(mask)} inliers ({inlier_ratio:.0%})")
    if inlier_ratio < 0.5:
        print("warning: fewer than half the correspondences are inliers; the hand "
              "may have been paired across views incorrectly", file=sys.stderr)

    # recoverPose resolves the four-fold decomposition ambiguity by requiring
    # triangulated points to lie in front of both cameras, and returns a unit
    # translation -- the scale epipolar geometry cannot observe.
    _, rotation, unit_translation, _ = cv2.recoverPose(essential, left, right, k_matrix, mask=mask)
    translation = unit_translation.reshape(3) * baseline_m
    return rotation, translation, inlier_ratio


def reprojection_residual(
    left: np.ndarray, right: np.ndarray, k_matrix: np.ndarray,
    rotation: np.ndarray, translation: np.ndarray,
) -> float:
    """Triangulate the correspondences and report the RMS reprojection error."""

    projection_left = k_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
    projection_right = k_matrix @ np.hstack((rotation, translation.reshape(3, 1)))
    homogeneous = cv2.triangulatePoints(projection_left, projection_right, left.T, right.T)
    points = (homogeneous[:3] / homogeneous[3]).T

    errors = []
    for projection, observed in ((projection_left, left), (projection_right, right)):
        projected = points @ projection[:, :3].T + projection[:, 3]
        projected = projected[:, :2] / projected[:, 2:3]
        errors.append(np.linalg.norm(projected - observed, axis=1))
    return float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))


def write_calibration(
    path, ids: list[str], image_size: tuple[int, int], k_matrix: np.ndarray,
    rotation: np.ndarray, translation: np.ndarray, pairs: int,
    inlier_ratio: float, residual_px: float,
) -> None:
    calibration = {
        "version": 1,
        "reference_camera_id": ids[0],
        "cameras": {
            ids[0]: {
                "image_size": list(image_size),
                "K": k_matrix.tolist(),
                "R": np.eye(3).tolist(),
                "t": [0.0, 0.0, 0.0],
            },
            ids[1]: {
                "image_size": list(image_size),
                "K": k_matrix.tolist(),
                "R": rotation.tolist(),
                "t": translation.tolist(),
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            f"# Generated by autocalibrate_multiview_server.py from {pairs} hand poses.\n"
            f"# Essential-matrix inliers {inlier_ratio:.0%}, triangulation RMS "
            f"{residual_px:.2f}px.\n"
            "# Intrinsics are WiLoR's nominal pinhole model, not a lens calibration,\n"
            "# and no distortion is modelled.  Scale comes from the --baseline given\n"
            "# on the command line; re-measure and regenerate if the rig moves.\n"
        )
        yaml.safe_dump(calibration, handle, sort_keys=False, default_flow_style=None)


def main() -> None:
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", action="append", type=parse_camera_spec,
                        required=True, metavar="ID=DEVICE",
                        help="exactly two; the first becomes the rig origin")
    parser.add_argument("--baseline", type=float, required=True,
                        help="measured distance between the lens centres, in metres")
    parser.add_argument("--out", type=Path,
                        default=Path("../isoswarm/configs/calibration/hand_rig.yaml"))
    parser.add_argument("--frames", type=int, default=120,
                        help="paired frames to accumulate before solving")
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
    parser.add_argument("--calibrate-only", action="store_true",
                        help="write the calibration and exit instead of streaming")
    args = parser.parse_args()

    if len(args.camera) != 2:
        parser.error("supply exactly two --camera entries")
    ids = [camera_id for camera_id, _ in args.camera]
    if len(set(ids)) != 2:
        parser.error("camera ids must be unique")
    if not 0.0 < args.baseline < 5.0:
        parser.error("--baseline must be the lens separation in metres, e.g. 0.185")

    device = choose_device(args.device)
    print(f"Using device: {device}")
    model, model_cfg, detector, use_fp16 = load_pipeline(device, args.fast, args.faster)

    def infer(frame):
        return extract_hands(frame, model, model_cfg, detector, device,
                             args.rescale_factor, args.conf, fp16=use_fp16)

    try:
        cameras = open_cameras(args.camera, args.width, args.height, args.fps)
    except RuntimeError as error:
        sys.exit(str(error))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        check_cameras_see_something(cameras)
        image_size = (args.width, args.height)
        k_matrix = nominal_intrinsics(model_cfg, *image_size)
        print(f"nominal focal length: {k_matrix[0, 0]:.1f}px, baseline: "
              f"{args.baseline * 100:.1f}cm")

        left, right = collect_correspondences(
            cameras, infer, args.width, args.height, args.frames, args.preview)
        rotation, translation, inlier_ratio = solve_extrinsics(
            left, right, k_matrix, args.baseline)
        residual_px = reprojection_residual(left, right, k_matrix, rotation, translation)
        print(f"triangulation RMS: {residual_px:.2f}px")
        if residual_px > 5.0:
            print("warning: RMS above 5px -- isoswarm's default "
                  "max_reprojection_error_px would reject most fused points; "
                  "recalibrate with slower, more varied hand motion", file=sys.stderr)

        write_calibration(args.out, ids, image_size, k_matrix, rotation, translation,
                          len(left) // 21, inlier_ratio, residual_px)
        print(f"wrote {args.out}")

        if args.calibrate_only:
            return
        target = (args.host, args.port)
        print(f"\nStreaming WiLoR v2 bundles to udp://{args.host}:{args.port}")
        stream_bundles(cameras, infer, sock, target, args.preview)
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
