"""Vibed cumulative normalization with globally merged distributed batch moments."""
import torch
from torch import nn
import torch.distributed as dist

class RunningNorm(nn.Module):
    def __init__(self,dim,max_count=2e8,eps=1e-2):
        super().__init__()
        self.register_buffer("mean",torch.zeros(dim))
        self.register_buffer("var",torch.ones(dim))
        self.register_buffer("count",torch.zeros(()))
        self.max_count,self.eps=max_count,eps
    @torch.no_grad()
    def update(self,x):
        if self.count>=self.max_count:return
        x=x.detach().reshape(-1,x.shape[-1]).float()
        n=x.new_tensor(x.shape[0]);total=x.sum(0);squares=x.square().sum(0)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(n);dist.all_reduce(total);dist.all_reduce(squares)
        if n==0:return
        m=total/n;v=(squares/n-m.square()).clamp_min(0)
        count=self.count+n;delta=m-self.mean
        self.var.copy_((self.var*self.count+v*n+delta.square()*self.count*n/count)/count)
        self.mean.add_(delta*n/count);self.count.copy_(count)
    def std(self):return (self.var+self.eps**2).sqrt()
    def forward(self,x):return (x-self.mean)/self.std()
