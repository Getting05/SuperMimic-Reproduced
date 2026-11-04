import torch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.world_model.heads import analytic_observation, flatten_history_term_major

def library():
    lib=MotionLibrary.__new__(MotionLibrary)
    lib.device=lib.storage_device=torch.device("cpu");lib.dtype=torch.float32
    lib._ref=torch.zeros(3,71);lib._ref[:,3]=1
    lib._ref[:,0]=torch.arange(3);lib._ref[:,7:10]=2
    lib._ref[:,13:42]=3;lib._ref[:,42:71]=4
    lib.frame_clip=torch.zeros(3,dtype=torch.long);lib.clip_last=torch.tensor([2])
    return lib

def test_future_order_and_exact_last_frame_velocity():
    g=library().goal_window(torch.tensor([0]),10).reshape(10,73)
    assert g.shape==(10,73)
    torch.testing.assert_close(g[:2,0],torch.tensor([1.,2.]))
    torch.testing.assert_close(g[:2,9:12],torch.full((2,3),2.))
    assert not g[2:,9:15].any()
    torch.testing.assert_close(g[:,15:44],torch.full((10,29),3.))
    torch.testing.assert_close(g[:2,44:],torch.full((2,29),.4))
    assert not g[2:,44:].any()

def test_obs_fields_and_time_major():
    s=library()._ref[:2];xi=torch.zeros(2,33);xi[:,1:30]=.01
    a=torch.full((2,29),5.)
    o=analytic_observation(s,a,xi,torch.ones(29),s[:,3:7])
    assert o.shape==(2,99)
    torch.testing.assert_close(o[:,:29],torch.full((2,29),1.99))
    torch.testing.assert_close(o[:,64:93],a)
    h=torch.stack([o+i for i in range(10)])
    flat=flatten_history_term_major(h)
    assert flat.shape==(2,990)
    torch.testing.assert_close(flat.reshape(2,10,99),h.permute(1,0,2))
