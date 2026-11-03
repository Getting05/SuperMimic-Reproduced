"""Astro P2 position PD with motor torque-speed envelopes."""

from dataclasses import dataclass, replace

import torch
from mjlab.actuator import IdealPdActuator, IdealPdActuatorCfg
from mjlab.actuator.pd_actuator import pd_torque


@dataclass(kw_only=True)
class P2ActuatorCfg(IdealPdActuatorCfg):
    joint_effort_limits: tuple[float, ...]
    joint_speed_limits: tuple[float, ...]
    motor_curve: tuple[float, float, float]  # knee speed, no-load speed, peak torque
    parallel: bool = False

    def build(self, entity, target_ids, target_names):
        return P2Actuator(self, entity, target_ids, target_names)


class P2Actuator(IdealPdActuator[P2ActuatorCfg]):
    def initialize(self, mj_model, model, data, device):
        super().initialize(mj_model, model, data, device)
        self.joint_effort_limits = torch.tensor(self.cfg.joint_effort_limits, device=device)
        self.joint_speed_limits = torch.tensor(self.cfg.joint_speed_limits, device=device)

    def apply_delay(self, cmd):
        if self._delay_buffer is None:
            return cmd
        self._delay_buffer.append(cmd.position_target)
        return replace(cmd, position_target=self._delay_buffer.compute())

    def compute(self, cmd):
        torque = pd_torque(self.stiffness, self.damping, cmd)
        return limited_torque(
            torque, cmd.vel, self.joint_effort_limits, self.joint_speed_limits,
            self.cfg.motor_curve, self.cfg.parallel,
        )


def _sum_difference(value: torch.Tensor) -> torch.Tensor:
    pitch, roll = value.unbind(-1)
    return torch.stack((pitch + roll, pitch - roll), dim=-1)


def limited_torque(torque, velocity, effort_limits, speed_limits, curve, parallel=False):
    """Limit motor torque and suppress drive above the joint speed limit.

    The parallel transmission is a fixed-ratio approximation: motor
    velocities are J qdot, and joint torque is J.T motor_torque. J has rows
    (1, 1) and (1, -1), giving the two-motor gain multiplier in the attachment.
    """
    knee, no_load, peak = curve
    motor_speed = _sum_difference(velocity) if parallel else velocity
    motor_torque = 0.5 * _sum_difference(torque) if parallel else torque
    limit = peak * ((no_load - motor_speed.abs()) / (no_load - knee)).clamp(0, 1)
    motor_torque = motor_torque.clamp(-limit, limit)
    torque = _sum_difference(motor_torque) if parallel else motor_torque

    # Apply a shared scale to a pair so joint-side clipping cannot violate
    # either motor's envelope. Counter-torque remains available at overspeed.
    driving_overspeed = (velocity.abs() >= speed_limits) & (torque * velocity > 0)
    scale = (effort_limits / torque.abs().clamp_min(1e-8)).clamp(max=1)
    scale = scale.masked_fill(driving_overspeed, 0)
    if parallel:
        scale = scale.amin(dim=-1, keepdim=True)
    return torque * scale
