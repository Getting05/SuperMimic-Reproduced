from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from supermimic.envs.mdp.commands import MultiClipMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def goal_future_frames(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """(N, K*68) future reference-frame features (plan D5)."""
    command = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return command.goal

def policy_frame(env, command_name="motion", include_ref_rot6d=True):
    from supermimic.envs.state_io import StateIO
    from supermimic.world_model.heads import analytic_observation
    robot=env.scene["robot"]
    cmd=env.command_manager.get_term(command_name)
    xi=torch.zeros(env.num_envs,33,device=env.device)
    xi[:,1:30]=robot.data.encoder_bias
    return analytic_observation(StateIO(env).read(),env.action_manager.action,xi,
                                robot.data.default_joint_pos,cmd.root_quat_w,include_ref_rot6d)

def policy_frame_noisy(env, command_name="motion", include_ref_rot6d=True):
    o=policy_frame(env,command_name,include_ref_rot6d)
    half=o.new_tensor([.01]*29+[.5]*29+[.2]*3+[.05]*3+[0.]*29+[.05]*6)
    return o+(torch.rand_like(o)*2-1)*half[:o.shape[-1]]
