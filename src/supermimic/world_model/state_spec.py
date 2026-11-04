"""State representation for the world model (plan D1–D3).

Model state s = [root_pos(3), root_quat wxyz(4), root_lin_vel_w(3),
                 root_ang_vel_w(3), joint_pos(29), joint_vel(29)] (71, origin-free)

Canonical features (network input, heading-invariant, 71):
    [root_z(1), rot6d(heading^-1 ⊗ root_quat)(6), root_lin_vel in heading frame(3),
     root_ang_vel_b(3), joint_pos(29), joint_vel(29)]

Network output ``delta`` (70) and the integration step (dt = 0.02 s):
    dv_h(3), dw_b(3), dqd(29)            velocity increments, heading / body frame
    dp_h(3), dth_b(3), dq(29)            position corrections on top of explicit integration
    v'  = v + R_h dv_h               w' = w + dw_b          qd' = qd + dqd
    p'  = p + dt v' + R_h dp_h       q' = q ⊗ exp(dt w' + dth_b)    qj' = qj + dt qd' + dq
With zero output the model extrapolates at constant velocity (a physical prior); the
last layer of the dynamics MLP is zero-initialised so training starts from it.

Loss / error normalisation (D3): per-quantity nominal scales sigma applied to the
70-d error vector [dp(3), rotvec(3), dv(3), dw(3), dq(29), dqd(29)].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from supermimic.kinematics.rotations import (
    heading_quat,
    quat_apply,
    quat_apply_inverse,
    quat_canonical,
    quat_from_rotvec,
    quat_mul,
    quat_normalize,
    rot6d_from_quat,
    rotvec_from_quat,
)

STATE_DIM = 71
FEAT_DIM = 71
DELTA_DIM = 70
NJ = 29


@dataclass
class StateSpecCfg:
    dt: float = 0.02
    # Nominal error scales (D3) for [root pos m, root rotvec rad, root lin vel m/s, root ang vel rad/s, joint pos rad, joint vel rad/s].
    err_scale: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0, 0.1, 2.0)
    # Output scales of the dynamics head for [dv_h, dw_b, dqd, dp_h, dth_b, dq].
    out_scale: tuple[float, ...] = (0.3, 0.5, 2.0, 0.01, 0.02, 0.02)
    # Input feature scales for [root_z, rot6d, lin_vel_h, ang_vel_b, joint_pos, joint_vel] (divide by these).
    in_scale: tuple[float, ...] = (1.0, 1.0, 1.0, 2.0, 1.0, 5.0)
    explicit_integration: bool = True
    # xi normalisation: (offset, scale) for friction, encoder bias, com offset.
    xi_offset: tuple[float, ...] = field(default_factory=lambda: (0.75,) + (0.0,) * 29 + (0.0,) * 3)
    xi_scale: tuple[float, ...] = field(default_factory=lambda: (0.45,) + (0.01,) * 29 + (0.05,) * 3)


class StateSpec:
    def __init__(self, cfg: StateSpecCfg, device, dtype=torch.float32):
        self.cfg = cfg
        self.dt = cfg.dt
        rep = lambda scales, sizes: torch.cat([torch.full((n,), float(s)) for s, n in zip(scales, sizes)])  # noqa: E731
        self.err_scale = rep(cfg.err_scale, (3, 3, 3, 3, NJ, NJ)).to(device=device, dtype=dtype)
        self.out_scale = rep(cfg.out_scale, (3, 3, NJ, 3, 3, NJ)).to(device=device, dtype=dtype)
        self.in_scale = rep(cfg.in_scale, (1, 6, 3, 3, NJ, NJ)).to(device=device, dtype=dtype)
        self.xi_offset = torch.tensor(cfg.xi_offset, device=device, dtype=dtype)
        self.xi_scale = torch.tensor(cfg.xi_scale, device=device, dtype=dtype)

    # ------------------------------------------------------------------ split / features

    @staticmethod
    def split(s: torch.Tensor):
        return s[..., 0:3], s[..., 3:7], s[..., 13:42], s[..., 7:10], quat_apply_inverse(s[..., 3:7], s[..., 10:13]), s[..., 42:71]

    def canonical(self, s: torch.Tensor) -> torch.Tensor:
        """Heading-invariant, scaled network input features (…, 71)."""
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        hq = heading_quat(q)
        q_local = quat_canonical(quat_mul(quat_apply_inverse_quat(hq), q))
        v_h = quat_apply_inverse(q, v)
        feat = torch.cat([p[..., 2:3], rot6d_from_quat(q_local), v_h, w, qj, qd], dim=-1)
        return feat

    def normalize_xi(self, xi: torch.Tensor) -> torch.Tensor:
        return (xi - self.xi_offset) / self.xi_scale

    # ------------------------------------------------------------------ integration

    def integrate(self, s: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Integrate direct frame-local increments; no constant-velocity prior."""
        p,q,qj,v,_,qd=self.split(s)
        w=s[...,10:13]
        dp,dth,dv,dw,dq,dqd=delta.split((3,3,3,3,29,29),-1)
        q_new=quat_normalize(quat_mul(q,quat_from_rotvec(dth)))
        return torch.cat((p+quat_apply(heading_quat(q),dp),q_new,
            v+quat_apply(q,dv),w+quat_apply(q,dw),qj+dq,qd+dqd),-1)

    def targets(self,s,s1):
        p,q,j,v,_,jd=self.split(s)
        p1,q1,j1,v1,_,jd1=self.split(s1)
        return torch.cat((quat_apply_inverse(heading_quat(q),p1-p),
            rotvec_from_quat(quat_mul(quat_apply_inverse_quat(q),q1)),
            quat_apply_inverse(q,v1-v),quat_apply_inverse(q,s1[...,10:13]-s[...,10:13]),
            j1-j,jd1-jd),-1)

    # ------------------------------------------------------------------ errors

    def error_vector(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        return self.targets(s_true,s_pred)/self.err_scale

    def normalized_sq_error(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Mean over dims of the squared normalised error (the reported 'prediction error')."""
        e = self.error_vector(s_pred, s_true)
        return (e * e).mean(-1)


def quat_apply_inverse_quat(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a unit quaternion (inverse)."""
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q.dtype, device=q.device)
