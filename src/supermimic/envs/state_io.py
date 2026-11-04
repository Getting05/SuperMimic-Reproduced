"""MuJoCo native state adapters for the origin-free Vibed 71D model state."""
from __future__ import annotations
import torch
from supermimic.kinematics.rotations import quat_apply, quat_apply_inverse, quat_normalize
STATE_DIM, NQ, NV = 71, 36, 35

def from_native(qpos, qvel, origins=None):
    """[p,quat,q] + [v,w_body,qd] -> [p,quat,v,w_world,q,qd]."""
    q = quat_normalize(qpos[...,3:7])
    p = qpos[...,:3] if origins is None else qpos[...,:3]-origins
    return torch.cat((p,q,qvel[...,:3],quat_apply(q,qvel[...,3:6]),
                      qpos[...,7:36],qvel[...,6:35]),-1)

def to_native(state, origins=None):
    """Model state -> native MuJoCo arrays, including world/body angular transform."""
    q = quat_normalize(state[...,3:7])
    p = state[...,:3] if origins is None else state[...,:3]+origins
    return (torch.cat((p,q,state[...,13:42]),-1),
            torch.cat((state[...,7:10],quat_apply_inverse(q,state[...,10:13]),
                       state[...,42:71]),-1))

class StateIO:
    def __init__(self,env):
        self.env,self.sim=env,env.sim
        self.origins=env.scene.env_origins.clone()
        assert self.sim.data.qpos.shape[-1]==NQ and self.sim.data.qvel.shape[-1]==NV
    def read(self):
        return from_native(self.sim.data.qpos[:],self.sim.data.qvel[:],self.origins)
    @torch.no_grad()
    def write(self,env_ids,state,forward=True):
        state=state.to(self.sim.data.qpos.dtype)
        qpos,qvel=to_native(state,self.origins[env_ids])
        self.sim.data.qpos[env_ids]=qpos
        self.sim.data.qvel[env_ids]=qvel
        if hasattr(self.sim.data,"qacc_warmstart"):
            self.sim.data.qacc_warmstart[env_ids]=0.
        if forward:
            self.sim.forward()
