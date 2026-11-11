"""Time-major vectorized transition ring. One row per complete env.step."""
from dataclasses import dataclass
import torch

@dataclass
class WindowBatch:
    states: torch.Tensor
    a_exec: torch.Tensor
    xi: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    frame: torch.Tensor
    mask: torch.Tensor

@dataclass
class StartBatch:
    state: torch.Tensor
    xi: torch.Tensor
    frame: torch.Tensor
    prev_action: torch.Tensor
    hist_states: torch.Tensor
    hist_actions: torch.Tensor
    hist_frames: torch.Tensor
    hist_xi: torch.Tensor

class ReplayBuffer:
    def __init__(self,num_envs,capacity,state_dim,act_dim,xi_dim,device):
        self.E,self.T=num_envs,capacity
        self.device=torch.device(device)
        self.ptr=self.rows=0
        def alloc(d=(),dtype=torch.float32):
            return torch.zeros((capacity,num_envs,*d),device=device,dtype=dtype)
        self.s=alloc((state_dim,));self.s1=alloc((state_dim,))
        self.a=alloc((act_dim,));self.a_prev=alloc((act_dim,))
        self.xi=alloc((xi_dim,));self.r=alloc()
        self.f=alloc(dtype=torch.int64);self.ep_step=alloc(dtype=torch.int32)
        self.done=alloc(dtype=torch.bool)
    @property
    def size(self):return min(self.rows,self.T)*self.E
    @torch.no_grad()
    def add(self,**transition):
        if set(transition)!=set(("s","s1","a","a_prev","xi","r","f","ep_step","done")):
            raise ValueError("Expected one complete vectorized transition")
        for name,value in transition.items():
            target=getattr(self,name)[self.ptr]
            if target.shape!=value.shape:raise ValueError(f"{name}: {value.shape} != {target.shape}")
            target.copy_(value)
        self.ptr=(self.ptr+1)%self.T;self.rows+=1
    def _starts(self,n,k=1):
        count=min(self.rows,self.T)
        if count<k:raise RuntimeError("Not enough replay rows")
        age=torch.randint(k-1,count,(n,),device=self.device)
        env=torch.randint(self.E,(n,),device=self.device)
        return (self.ptr-1-age)%self.T,env,age
    @torch.no_grad()
    def sample_windows(self,n,k):
        t,e,_=self._starts(n,k)
        tt=(t[None]+torch.arange(k,device=self.device)[:,None])%self.T
        done=self.done[tt,e]
        mask=torch.cat((torch.ones_like(done[:1]),(~done[:-1]).cumprod(0).bool()),0)
        states=torch.cat((self.s[t,e][None],self.s1[tt,e]),0)
        return WindowBatch(states,self.a[tt,e],self.xi[t,e],self.r[tt,e],
                           done,self.f[tt,e],mask)
    @torch.no_grad()
    def sample_starts(self,n,hist):
        t,e,age=self._starts(n)
        count=min(self.rows,self.T)
        available=torch.minimum(self.ep_step[t,e].long(),count-1-age)
        back=torch.arange(hist-1,-1,-1,device=self.device)[:,None]
        tt=(t[None]-torch.minimum(back,available[None]))%self.T
        return StartBatch(self.s[t,e],self.xi[t,e],self.f[t,e],self.a_prev[t,e],
            self.s[tt,e],self.a_prev[tt,e],self.f[tt,e],self.xi[tt,e])
