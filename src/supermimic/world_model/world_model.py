"""World model: learned dynamics + learned termination head + analytic observation/reward heads."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from supermimic.policy.normalization import RunningNorm
from supermimic.world_model.dynamics import DynamicsMLP
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg


@dataclass
class WorldModelCfg:
    hidden: tuple[int, ...] = (2048, 2048, 2048)
    term_hidden: tuple[int, ...] = (64, 64)
    state_spec: StateSpecCfg = field(default_factory=StateSpecCfg)
    term_loss_weight: float = 1.0
    height_threshold: float = 0.25
    tilt_threshold: float = 0.8


class WorldModel(nn.Module):
    def __init__(self, cfg: WorldModelCfg, heads: AnalyticHeads, device):
        super().__init__()
        self.cfg = cfg
        self.spec = StateSpec(cfg.state_spec, device)
        self.dynamics = DynamicsMLP(hidden=cfg.hidden)
        self.in_norm=RunningNorm(133)
        self.out_norm=RunningNorm(70,eps=1e-4)
        self.register_buffer("err_floor",torch.tensor([.003]*3+[.005]*3+[.02]*3+[.05]*3+[.003]*29+[.05]*29))
        self.heads = heads  # analytic observation / reward / FK (no parameters)
        self.to(device)

    # ------------------------------------------------------------------ one step

    def step(self, s: torch.Tensor, action: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """s_{t+1} = integrate(s_t, f(s_t, a_t, xi))."""
        x=self.inputs(s,action,xi)
        # Only network matmuls use autocast. Integration remains float32.
        use_amp=s.is_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast("cuda" if s.is_cuda else "cpu",dtype=torch.bfloat16,enabled=use_amp):
            pred=self.dynamics.mlp(self.in_norm(x))
        delta=pred.float()*self.out_norm.std()+self.out_norm.mean
        return self.spec.integrate(s.float(),delta)

    def inputs(self,s,a,xi):
        return torch.cat((self.spec.canonical(s),a,self.spec.normalize_xi(xi)),-1)

    @torch.no_grad()
    def update_normalization(self,s,s1,a,xi):
        self.in_norm.update(self.inputs(s,a,xi))
        self.out_norm.update(self.spec.targets(s,s1))

    def enable_compile(self,mode="default"):
        self._step_fn=torch.compile(self.step,mode=None if mode=="default" else mode,dynamic=False) if mode else self.step

    def window_loss(self,states,a_exec,xi,done,refs,mask=None):
        k=a_exec.shape[0];s=states[0];loss=s.new_zeros(());errors=[]
        std=torch.maximum(self.out_norm.std(),self.err_floor)
        step=getattr(self,"_step_fn",self.step)
        for j in range(k):
            s=step(s,a_exec[j],xi)
            e=(self.spec.targets(states[j+1],s)/std).square().mean(-1)
            valid=mask[j] if mask is not None else torch.ones_like(e)
            loss=loss+(e*valid).mean()
            errors.append((e*valid).sum()/valid.sum().clamp_min(1))
        return loss,dict(state_loss=loss.detach()/k,nse_1step=errors[0].detach(),
                         nse_kstep=errors[-1].detach())

    # ------------------------------------------------------------------ diagnostics

    @torch.no_grad()
    def rollout_error(self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor, mask=None) -> torch.Tensor:
        """Open-loop k-step normalised squared error per step (k,), averaged over the batch."""
        k = a_exec.shape[0]
        s = states[0]
        out = []
        for j in range(k):
            s = self.step(s, a_exec[j], xi)
            e=(self.spec.targets(states[j+1],s)/torch.maximum(self.out_norm.std(),self.err_floor)).square().mean(-1)
            valid=mask[j] if mask is not None else torch.ones_like(e)
            out.append((e*valid).sum()/valid.sum().clamp_min(1))
        return torch.stack(out)
