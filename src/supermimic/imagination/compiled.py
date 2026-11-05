"""One imagined step with fused analytic FK/reward/termination heads.

The WM/autocast and policy-feedback observation retain their original execution
paths. Only analytic heads are compiled into one graph to reduce kernel launches.
Their body positions are also returned for critic reuse without repeating FK.
Reference tensors are gathered in advance, keeping compiled shapes static.
"""

from __future__ import annotations

import torch

from supermimic.world_model.heads import ReferenceFrame, analytic_observation
from supermimic.world_model.world_model import WorldModel

REWARD_TERM_NAMES = (
    "motion_global_root_pos",
    "motion_global_root_ori",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
    "action_rate_l2",
    "joint_limit",
)


class ImagStep:
    def __init__(self, wm: WorldModel, default_joint_pos: torch.Tensor, compile_mode: str | None = None):
        self.wm = wm
        self.q0 = default_joint_pos
        self._fn = self._heads
        if compile_mode:
            mode = None if compile_mode == "default" else compile_mode
            # Preserve eager rounding/casts and libdevice math while fusing launches.
            # In particular, do not change the WM's BF16 execution path.
            options = dict(torch._inductor.list_mode_options(mode or "default"))
            options["emulate_precision_casts"] = True
            self._fn = torch.compile(self._heads, options=options, dynamic=False, fullgraph=True)

    def _step(self, s, a, a_prev, xi, ref_body_pos, ref_body_quat, ref_body_lin, ref_body_ang, ref_root_pos, ref_root_quat):
        # Keep the WM's existing autocast/LayerNorm execution and rounding.
        s_next = self.wm.step(s, a, xi)
        state, r, terms, termination, anchor, bodies = self._fn(
            s_next, a, a_prev, xi, ref_body_pos, ref_body_quat,
            ref_body_lin, ref_body_ang, ref_root_pos, ref_root_quat)
        # Keep the policy feedback observation on its original eager path too:
        # tiny geometry roundoff must not feed back into the BF16 WM rollout.
        obs = analytic_observation(s_next, a, xi, self.q0, ref_root_quat)
        return state, r, terms, termination, obs, anchor, bodies

    def _heads(self, s_next, a, a_prev, xi, ref_body_pos, ref_body_quat, ref_body_lin, ref_body_ang, ref_root_pos, ref_root_quat):
        ref = ReferenceFrame(ref_body_pos, ref_body_quat, ref_body_lin, ref_body_ang, ref_root_pos, ref_root_quat)
        bodies = self.wm.heads.robot_bodies(s_next)
        terms = self.wm.heads.reward(s_next, a, a_prev, ref, bodies)
        r = terms["total"]
        term_vec = torch.stack([terms[k] for k in REWARD_TERM_NAMES], dim=0)  # (8, n)
        termination = self.wm.heads.soft_termination(s_next, ref)
        ai = self.wm.heads.anchor_index
        anchor_pose = torch.cat([bodies["pos"][..., ai, :], bodies["quat"][..., ai, :]], dim=-1)  # (n, 7)
        return s_next, r, term_vec, termination, anchor_pose, bodies["pos"]

    def __call__(self, s, a, a_prev, xi, ref: ReferenceFrame, *, return_body_pos: bool = False):
        outputs = self._step(s, a, a_prev, xi, ref.body_pos_w, ref.body_quat_w, ref.body_lin_vel_w, ref.body_ang_vel_w, ref.root_pos_w, ref.root_quat_w)
        return outputs if return_body_pos else outputs[:6]
