"""Analytic BeyondMimic tracking reward in torch (plan D4/D7).

One differentiable implementation shared by the simulator side (tests / logging)
and the imagination side (world-model reward head). Mirrors mjlab's
``tasks/tracking/mdp/rewards.py`` term by term, including the re-anchoring of the
reference bodies to the robot anchor in xy + yaw only
(``MotionCommand.update_relative_body_poses``).

Inputs are world-frame link quantities of the *tracked* bodies (same ordering as
``TRACKED_BODIES``), the anchor being one of them (``anchor_index``).
Rewards are returned already weighted and multiplied by ``dt`` like mjlab's reward
manager (``scale_rewards_by_dt=True``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_box_minus, quat_inv, quat_mul, yaw_quat


@dataclass
class TrackingRewardCfg:
    std_body_pos: float = 0.3
    std_body_ori: float = 0.4
    std_body_lin_vel: float = 1.0
    std_body_ang_vel: float = 3.14
    std_root_pos: float = 0.3
    std_root_ori: float = 0.4
    w_body_pos: float = 1.0
    w_body_ori: float = 1.0
    w_body_lin_vel: float = 1.0
    w_body_ang_vel: float = 1.0
    w_root_pos: float = 0.5
    w_root_ori: float = 0.5
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    dt: float = 0.02

    @property
    def max_step_reward(self) -> float:
        return self.dt * (self.w_body_pos + self.w_body_ori + self.w_body_lin_vel + self.w_body_ang_vel + self.w_root_pos + self.w_root_ori)


def reanchor_reference(
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
    ref_anchor_pos_w: torch.Tensor,
    ref_anchor_quat_w: torch.Tensor,
    robot_anchor_pos_w: torch.Tensor,
    robot_anchor_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express the reference bodies in a frame that shares the robot anchor's xy position and yaw.

    Exactly mjlab's ``update_relative_body_poses``: z, roll and pitch stay those of the reference.
    Shapes: bodies (N, B, ·), anchors (N, ·).
    """
    nb = ref_body_pos_w.shape[-2]
    delta_pos = robot_anchor_pos_w.clone()
    delta_pos = torch.cat([delta_pos[..., :2], ref_anchor_pos_w[..., 2:3]], dim=-1)  # (N,3)
    delta_ori = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(ref_anchor_quat_w)))  # (N,4)
    delta_ori_b = delta_ori[..., None, :].expand(*delta_ori.shape[:-1], nb, 4)
    rel_quat = quat_mul(delta_ori_b, ref_body_quat_w)
    rel_pos = delta_pos[..., None, :] + quat_apply(delta_ori_b, ref_body_pos_w - ref_anchor_pos_w[..., None, :])
    return rel_pos, rel_quat


def _sq_quat_err(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Squared geodesic angle between quaternions, smooth at zero (== quat_error_magnitude**2)."""
    v = quat_box_minus(q1, q2)
    return (v * v).sum(-1)


def tracking_reward_terms(
    *,
    robot_body_pos_w: torch.Tensor,
    robot_body_quat_w: torch.Tensor,
    robot_body_lin_vel_w: torch.Tensor,
    robot_body_ang_vel_w: torch.Tensor,
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
    ref_body_lin_vel_w: torch.Tensor,
    ref_body_ang_vel_w: torch.Tensor,
    anchor_index: int,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    joint_pos: torch.Tensor,
    soft_limits: torch.Tensor,  # (29, 2) or (N, 29, 2)
    cfg: TrackingRewardCfg,
) -> dict[str, torch.Tensor]:
    """Per-term weighted, dt-scaled rewards (N,) plus 'total'."""
    ra_pos = robot_body_pos_w[..., anchor_index, :]
    ra_quat = robot_body_quat_w[..., anchor_index, :]
    ma_pos = ref_body_pos_w[..., anchor_index, :]
    ma_quat = ref_body_quat_w[..., anchor_index, :]
    rel_pos, rel_quat = reanchor_reference(ref_body_pos_w, ref_body_quat_w, ma_pos, ma_quat, ra_pos, ra_quat)

    e_root_pos = ((ma_pos - ra_pos) ** 2).sum(-1)
    e_root_ori = _sq_quat_err(ma_quat, ra_quat)
    e_body_pos = ((rel_pos - robot_body_pos_w) ** 2).sum(-1).mean(-1)
    e_body_ori = _sq_quat_err(rel_quat, robot_body_quat_w).mean(-1)
    e_body_lin = ((ref_body_lin_vel_w - robot_body_lin_vel_w) ** 2).sum(-1).mean(-1)
    e_body_ang = ((ref_body_ang_vel_w - robot_body_ang_vel_w) ** 2).sum(-1).mean(-1)

    lo = soft_limits[..., 0]
    hi = soft_limits[..., 1]
    out = (-(joint_pos - lo)).clamp(min=0.0) + (joint_pos - hi).clamp(min=0.0)
    joint_limit = out.sum(-1)
    action_rate = ((action - prev_action) ** 2).sum(-1)

    dt = cfg.dt
    terms = {
        "motion_global_root_pos": dt * cfg.w_root_pos * torch.exp(-e_root_pos / cfg.std_root_pos**2),
        "motion_global_root_ori": dt * cfg.w_root_ori * torch.exp(-e_root_ori / cfg.std_root_ori**2),
        "motion_body_pos": dt * cfg.w_body_pos * torch.exp(-e_body_pos / cfg.std_body_pos**2),
        "motion_body_ori": dt * cfg.w_body_ori * torch.exp(-e_body_ori / cfg.std_body_ori**2),
        "motion_body_lin_vel": dt * cfg.w_body_lin_vel * torch.exp(-e_body_lin / cfg.std_body_lin_vel**2),
        "motion_body_ang_vel": dt * cfg.w_body_ang_vel * torch.exp(-e_body_ang / cfg.std_body_ang_vel**2),
        "action_rate_l2": dt * cfg.w_action_rate * action_rate,
        "joint_limit": dt * cfg.w_joint_limit * joint_limit,
    }
    total = terms["motion_global_root_pos"]
    for k, v in terms.items():
        if k != "motion_global_root_pos":
            total = total + v
    terms["total"] = total
    return terms

def simulator_heads(env):
    """Shared analytic FK on post-integration qpos, avoiding stale mj_step derived poses."""
    if not hasattr(env, "_parity_heads"):
        from supermimic.kinematics.g1_fk import G1Kinematics
        from supermimic.world_model.heads import AnalyticHeads
        cmd=env.command_manager.get_term("motion")
        robot=env.scene["robot"]
        # Use the unmerged robot model: scene origins must not enter analytic FK.
        if cmd.cfg.robot_variant == "p2":
            from supermimic.envs.p2 import load_p2_model
            model=load_p2_model()
        else:
            from supermimic.data.motion_convert import load_g1_model
            model=load_g1_model()
        env._parity_heads=AnalyticHeads(G1Kinematics(model,device=env.device),
            cmd.cfg.body_names,"pelvis","pelvis",robot.data.soft_joint_pos_limits[0],
            robot.data.default_joint_pos[0])
    return env._parity_heads

def next_reference(env, *, bodies=True):
    from supermimic.world_model.wm_trainer import reference_at
    cmd=env.command_manager.get_term("motion")
    frame = torch.minimum(cmd.frame + 1, cmd.library.last_frame_of(cmd.frame))
    if bodies:
        return reference_at(cmd.library, frame)
    from supermimic.world_model.heads import ReferenceFrame
    ref = cmd.library.ref(frame)
    return ReferenceFrame(None, None, None, None, ref.root_pos_w, ref.root_quat_w)

def simulator_reward_term(env, term, weight, reuse=False):
    from supermimic.envs.state_io import StateIO
    # RewardManager visits terms in insertion order. The first nonzero term
    # refreshes the cache on EVERY compute() call, including repeated calls or
    # manual resets in the same simulator step. Later terms only read it.
    step = getattr(env, "common_step_counter", None)
    if not reuse or getattr(env, "_parity_reward_step", None) != step or not hasattr(env, "_parity_reward_terms"):
        heads = simulator_heads(env)
        env._parity_reward_terms = heads.reward(
            StateIO(env).read(), env.action_manager.action,
            env.action_manager.prev_action, next_reference(env))
        env._parity_reward_step = step
    return env._parity_reward_terms[term] / (weight * env.step_dt)

def simulator_termination_error(env):
    from supermimic.envs.state_io import StateIO
    s=StateIO(env).read()
    return simulator_heads(env).termination_features(s[:,:3],s[:,3:7],next_reference(env, bodies=False))
