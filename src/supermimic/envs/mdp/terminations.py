"""Paper terminations: root (pelvis) height deviation and tilt; clip end as time-out."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from supermimic.envs.mdp.commands import MultiClipMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def bad_root_height(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.25) -> torch.Tensor:
    from supermimic.envs.mdp.rewards import simulator_termination_error
    return simulator_termination_error(env)[:,0].abs() > threshold


def bad_root_tilt(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.8) -> torch.Tensor:
    from supermimic.envs.mdp.rewards import simulator_termination_error
    return simulator_termination_error(env)[:,1].abs() > threshold


def clip_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return cmd.clip_end


def sgs_horizon(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return cmd.record_sgs_step()
