import torch
from torch import nn
from supermimic.policy.normalization import RunningNorm
from supermimic.policy.nets import Policy,PolicyCfg,Critic,CriticCfg

def test_norm_moments_and_checkpoint():
    n=RunningNorm(3)
    x=torch.randn(31,3)
    n.update(x[:12]);n.update(x[12:])
    torch.testing.assert_close(n.mean,x.mean(0))
    torch.testing.assert_close(n.var,x.var(0,unbiased=False))
    assert n.count==31
    m=RunningNorm(3);m.load_state_dict(n.state_dict())
    torch.testing.assert_close(m(x),n(x))

def test_fsq_elu_dims_grad():
    p=Policy(PolicyCfg(policy_obs_ref_rot6d=True,fsq=True,encoder_hidden=(32,),decoder_hidden=(32,)))
    x=torch.randn(3,730,requires_grad=True)
    z=p.encoders["motion"](x)
    assert z.shape==(3,40) and torch.equal(z,z.round()) and z.abs().max()<=4
    a=p(torch.randn(3,10,99),x)
    assert a.shape==(3,29)
    a.sum().backward();assert torch.isfinite(x.grad).all()
    assert any(isinstance(m,nn.ELU) for m in p.modules())
    c=Critic(CriticCfg());assert c(torch.randn(3,184),torch.randn(3,730)).shape==(3,)
    assert c.norm.mean.numel()==914
