import torch
from supermimic.world_model.state_spec import StateSpec,StateSpecCfg
from supermimic.kinematics.rotations import quat_from_rotvec
from supermimic.world_model.world_model import WorldModel,WorldModelCfg

def test_direct_increment_closure_and_gradient():
    spec=StateSpec(StateSpecCfg(),"cpu")
    s=torch.randn(8,71);s[:,3:7]=quat_from_rotvec(torch.randn(8,3))
    d=(torch.randn(8,70)*.01).requires_grad_()
    s1=spec.integrate(s,d)
    torch.testing.assert_close(spec.targets(s,s1),d,atol=5e-7,rtol=1e-4)
    s1.sum().backward();assert torch.isfinite(d.grad).all()

def test_wm_dims_norm_masked_sum_and_no_learned_term():
    wm=WorldModel(WorldModelCfg(hidden=(32,32,32)),None,"cpu")
    assert not hasattr(wm,"term_head")
    s=torch.zeros(9,4,71);s[:,:,3]=1
    a=torch.zeros(8,4,29);xi=torch.zeros(4,33)
    wm.update_normalization(s[0],s[1],a[0],xi)
    assert wm.in_norm.mean.shape==(133,) and wm.out_norm.mean.shape==(70,)
    assert sum(isinstance(m,torch.nn.LayerNorm) for m in wm.modules())==3
    loss,m=wm.window_loss(s,a,xi,torch.zeros(8,4,dtype=torch.bool),None,torch.ones(8,4,dtype=torch.bool))
    assert torch.isfinite(loss)
    loss.backward()
