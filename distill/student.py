"""Depth-reduced ViT student for WiLoR, and the distillation losses.

Design constraint that drives everything: `wilor.py:39` hardcodes
`RefineNet(cfg, feat_dim=1280)` and RefineNet consumes a 1280x16x12 feature map,
with `out_dim = 1280//8 + 1280//4 + 1280//2`.  So the student **must keep width
1280**; narrowing to ViT-B would force RefineNet to be retrained too, which needs
the MANO supervision chain and is a far bigger job.

Depth, on the other hand, is free: a 16-layer / 1280-wide student is a drop-in
replacement for `model.backbone`, and RefineNet + both MANO passes stay frozen and
shared.  That also means:
  * the student can be *initialised from the teacher's own layers* rather than
    from scratch, and
  * intermediate hidden states can be matched layer-by-layer (student k <-> teacher
    2k+1), which is what makes this converge in hours instead of days.

Because RefineNet and MANO are frozen and shared, matching the ViT's outputs
reproduces the whole pipeline -- so distillation needs NO ground-truth labels, only
hand images.  The teacher generates every target.

Gotcha baked in here: `ViT.train()` (vit.py:407) is overridden without returning
self, so `.eval()` returns None.  Never chain `.half().to(dev).eval()`.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_student(teacher, depth=16, init="uniform"):
    """Clone the teacher and keep `depth` of its 32 blocks.

    init="uniform" spreads the kept layers evenly, which keeps the student's layer k
    aligned with teacher layer keep[k] for hidden-state matching.
    """
    n_teacher = len(teacher.blocks)
    if init == "uniform":
        keep = [round(i * (n_teacher - 1) / (depth - 1)) for i in range(depth)]
    elif init == "front":
        keep = list(range(depth))
    else:
        raise ValueError(init)

    student = copy.deepcopy(teacher)
    student.blocks = nn.ModuleList([copy.deepcopy(teacher.blocks[i]) for i in keep])
    student.skip_blocks = False
    student.blocks_to_skip = []
    return student, keep


def hidden_states(vit, x, want_layers=None):
    """Run the ViT and return (outputs, [hidden after each requested block]).

    Mirrors ViT.forward_features exactly; the upstream code has no hook for this.
    """
    B = x.shape[0]
    x, (Hp, Wp) = vit.patch_embed(x)
    if vit.pos_embed is not None:
        x = x + vit.pos_embed[:, 1:] + vit.pos_embed[:, :1]
    n_joints = vit.cfg.MANO.NUM_HAND_JOINTS + 1
    pose_tokens = vit.pose_emb(
        vit.init_hand_pose.reshape(1, n_joints, vit.joint_rep_dim)).repeat(B, 1, 1)
    shape_tokens = vit.shape_emb(vit.init_betas).unsqueeze(1).repeat(B, 1, 1)
    cam_tokens = vit.cam_emb(vit.init_cam).unsqueeze(1).repeat(B, 1, 1)
    h = torch.cat([pose_tokens, shape_tokens, cam_tokens, x], 1)

    keep = set(range(len(vit.blocks))) if want_layers is None else set(want_layers)
    states = []
    for i, blk in enumerate(vit.blocks):
        h = blk(h)
        if i in keep:
            states.append(h)

    h = vit.last_norm(h)
    pose_feat = h[:, :n_joints]
    shape_feat = h[:, n_joints:1 + n_joints]
    cam_feat = h[:, 1 + n_joints:2 + n_joints]
    pred_hand_pose = vit.decpose(pose_feat).reshape(B, -1) + vit.init_hand_pose
    pred_betas = vit.decshape(shape_feat).reshape(B, -1) + vit.init_betas
    pred_cam = vit.deccam(cam_feat).reshape(B, -1) + vit.init_cam
    feats = {"hand_pose": pred_hand_pose, "betas": pred_betas, "cam": pred_cam}
    img_feat = h[:, 2 + n_joints:].reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2)
    return feats, pred_cam, img_feat, states


def joints_from_backbone_out(model, feats, pred_cam, img_feat, focal_length):
    """Frozen RefineNet + both MANO passes -> joints, refined camera, projected 2D.

    This is `WiLoR.forward_step` minus the backbone.  It returns the *projected* 2D
    keypoints as well, computed exactly as wilor.py:134-156 does, because that is the
    quantity the downstream fusion consumes and it is the one that must be supervised:
    it depends on the RefineNet-refined camera, and depth is z = 2f/(IMAGE_SIZE*s),
    so a small error in the predicted scale s moves every projected landmark.
    """
    from wilor.utils.geometry import perspective_projection, rot6d_to_rotmat

    B = img_feat.shape[0]
    hp = rot6d_to_rotmat(feats["hand_pose"]).view(B, model.cfg.MANO.NUM_HAND_JOINTS + 1,
                                                  3, 3)
    temp_params = {"global_orient": hp[:, [0]].reshape(B, -1, 3, 3),
                   "hand_pose": hp[:, 1:].reshape(B, -1, 3, 3),
                   "betas": feats["betas"].reshape(B, -1)}
    temp_vertices = model.mano(**temp_params, pose2rot=False).vertices
    pred_params, cam = model.refine_net(img_feat, temp_vertices, pred_cam, feats,
                                        focal_length)
    out = model.mano(global_orient=pred_params["global_orient"].reshape(B, -1, 3, 3),
                     hand_pose=pred_params["hand_pose"].reshape(B, -1, 3, 3),
                     betas=pred_params["betas"].reshape(B, -1), pose2rot=False)
    joints = out.joints.reshape(B, -1, 3)

    img_size = model.cfg.MODEL.IMAGE_SIZE
    cam_t = torch.stack([cam[:, 1], cam[:, 2],
                         2 * focal_length[:, 0] / (img_size * cam[:, 0] + 1e-9)], dim=-1)
    kp2d = perspective_projection(joints, translation=cam_t,
                                  focal_length=focal_length / img_size)
    return joints, cam, kp2d.reshape(B, -1, 2)


class DistillLoss(nn.Module):
    """Feature + token + hidden-state + task-space distillation.

    Weights default to putting real mass on the task-space term: every earlier
    experiment showed the camera scale is the fragile quantity, and the fused 3D is
    2D-limited, so optimising the final joints directly is worth more than matching
    features alone.
    """

    def __init__(self, w_feat=1.0, w_tok=1.0, w_hidden=0.5, w_kp=5.0, w_cam=5.0,
                 w_kp2d=20.0):
        super().__init__()
        self.w_feat, self.w_tok = w_feat, w_tok
        self.w_hidden, self.w_kp, self.w_cam = w_hidden, w_kp, w_cam
        self.w_kp2d = w_kp2d

    @staticmethod
    def _nmse(a, b):
        """MSE normalised by the target's scale, so terms are comparable."""
        return F.mse_loss(a.float(), b.float()) / (b.float().pow(2).mean() + 1e-8)

    def forward(self, s, t):
        parts = {}
        parts["feat"] = self._nmse(s["img_feat"], t["img_feat"])
        parts["tok"] = (self._nmse(s["feats"]["hand_pose"], t["feats"]["hand_pose"])
                        + self._nmse(s["feats"]["betas"], t["feats"]["betas"]))
        parts["cam"] = self._nmse(s["feats"]["cam"], t["feats"]["cam"])
        if s.get("states") and t.get("states"):
            parts["hidden"] = sum(self._nmse(a, b)
                                  for a, b in zip(s["states"], t["states"])) / len(s["states"])
        else:
            parts["hidden"] = torch.zeros((), device=parts["feat"].device)
        if s.get("joints") is not None:
            # metres -> millimetres so the number is readable next to the eval metrics
            parts["kp"] = (s["joints"] - t["joints"]).norm(dim=-1).mean() * 1000.0
            parts["kp_rel"] = ((s["joints"] - s["joints"][:, :1])
                               - (t["joints"] - t["joints"][:, :1])).norm(dim=-1).mean() * 1000.0
        else:
            parts["kp"] = parts["kp_rel"] = torch.zeros((), device=parts["feat"].device)

        # THE term that matters.  kp_rel subtracts the wrist and therefore carries no
        # gradient at all on global translation / camera scale, while the projected 2D
        # landmarks -- what the fusion actually triangulates -- are dominated by the
        # refined camera.  Supervising only kp_rel made the student's articulation
        # improve while its 2D error got *worse* than the untrained initialisation
        # (11.17 px -> 12.98 px at step 2000).  Crop-space projection is in units of
        # the 256 px crop, so multiply by IMAGE_SIZE to read out as pixels.
        if s.get("kp2d") is not None:
            parts["kp2d_px"] = (s["kp2d"] - t["kp2d"]).norm(dim=-1).mean() * 256.0
        else:
            parts["kp2d_px"] = torch.zeros((), device=parts["feat"].device)

        total = (self.w_feat * parts["feat"] + self.w_tok * parts["tok"]
                 + self.w_cam * parts["cam"] + self.w_hidden * parts["hidden"]
                 + self.w_kp * parts["kp_rel"] / 10.0
                 + self.w_kp2d * parts["kp2d_px"] / 10.0)
        return total, {k: float(v) for k, v in parts.items()}
