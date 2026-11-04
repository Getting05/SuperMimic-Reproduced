"""mjlab RecorderTerm that streams simulator transitions into a ReplayBuffer.

Hook usage (see ``mjlab/managers/recorder_manager.py``):

* ``record_post_reset(ids)``  fresh episode start: remember s_0, xi, target frame.
* ``record_post_step()``      for envs that did not reset: write row (s_t, a_t, a_exec, xi, r, ...),
                              then remember s_{t+1} as the next row's state.
* ``record_pre_reset(ids)``   terminal step: write the last transition row plus a *terminal* row
                              holding s_{T+1} with its termination label; qpos/qvel are final here
                              even though derived kinematics are one substep stale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg

from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.envs.sim_context import XI_DIM, SimContext
from supermimic.envs.state_io import STATE_DIM, StateIO

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class ReplayRecorder(RecorderTerm):
    def __init__(self, cfg: ReplayRecorderCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self.state_io = StateIO(env)
        self.ctx = SimContext(env, p["foot_geom_regex"], p.get("com_body_name", "torso_link"))
        self.command_name = p.get("command_name", "motion")
        self.action_name = p.get("action_name", "joint_pos")
        n = env.num_envs
        self.buffer = ReplayBuffer(n, int(p["capacity"]), STATE_DIM, 29, XI_DIM, env.device)
        self.prev_s = torch.zeros(n,STATE_DIM,device=env.device)
        self.prev_xi = torch.zeros(n,XI_DIM,device=env.device)
        self.prev_frame = torch.zeros(n,dtype=torch.long,device=env.device)
        self.prev_action = torch.zeros(n,29,device=env.device)
        self.ep_step = torch.zeros(n,dtype=torch.int32,device=env.device)
        self.pending = {}
        self.enabled = True

    @property
    def command(self):
        return self._env.command_manager.get_term(self.command_name)

    def _remember(self,ids):
        self.prev_s[ids]=self.state_io.read()[ids]
        self.prev_xi[ids]=self.ctx.read()[ids]
        self.prev_frame[ids]=self.command.frame[ids]

    def record_post_reset(self,env_ids):
        if not self.enabled:return
        self._remember(env_ids)
        self.prev_action[env_ids]=0
        self.ep_step[env_ids]=0

    def record_pre_reset(self,env_ids):
        if not self.enabled or not env_ids.numel():return
        env=self._env
        # Snapshot original episode fields before reset hooks overwrite them.
        for name,value in dict(s=self.prev_s,s1=self.state_io.read(),
            a=env.action_manager.action,a_prev=self.prev_action,xi=self.prev_xi,
            r=env.reward_buf,f=self.prev_frame,ep_step=self.ep_step,done=env.reset_buf).items():
            if name not in self.pending:self.pending[name]=value.clone()
            self.pending[name][env_ids]=value[env_ids]

    def record_post_step(self):
        if not self.enabled:return
        env=self._env
        data=dict(s=self.prev_s.clone(),s1=self.state_io.read(),
            a=env.action_manager.action.clone(),a_prev=self.prev_action.clone(),
            xi=self.prev_xi.clone(),r=env.reward_buf.clone(),f=self.prev_frame.clone(),
            ep_step=self.ep_step.clone(),done=env.reset_buf.clone())
        ids=env.reset_buf.nonzero().flatten()
        for name,value in self.pending.items():
            data[name][ids]=value[ids]
        self.buffer.add(**data)
        self.pending.clear()
        active=(~env.reset_buf).nonzero().flatten()
        self.ep_step[active]+=1
        self.prev_action[active]=data["a"][active]
        self._remember(active)


@dataclass
class ReplayRecorderCfg(RecorderTermCfg):
    func: type = ReplayRecorder
    params: dict = field(default_factory=lambda: {"capacity": 1000, "foot_geom_regex": r"^(left|right)_foot[1-7]_collision$"})
