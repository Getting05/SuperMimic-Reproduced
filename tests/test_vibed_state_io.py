"""Numerical contracts for native MuJoCo/model state boundaries."""
from types import SimpleNamespace
import torch
from supermimic.envs.state_io import from_native, to_native, StateIO
from supermimic.kinematics.rotations import quat_from_rotvec, quat_apply

def test_native_roundtrip_and_explicit_fields():
    torch.manual_seed(71)
    qp=torch.randn(4,36,dtype=torch.float64)
    qp[:,3:7]=quat_from_rotvec(torch.randn(4,3,dtype=torch.float64))
    qv=torch.randn(4,35,dtype=torch.float64)
    origin=torch.randn(4,3,dtype=torch.float64)
    state=from_native(qp,qv,origin)
    torch.testing.assert_close(state[:,:3],qp[:,:3]-origin)
    torch.testing.assert_close(state[:,7:10],qv[:,:3])
    torch.testing.assert_close(state[:,10:13],quat_apply(qp[:,3:7],qv[:,3:6]))
    torch.testing.assert_close(state[:,13:42],qp[:,7:36])
    back=to_native(state,origin)
    torch.testing.assert_close(back[0],qp)
    torch.testing.assert_close(back[1],qv)

def test_stateio_partial_write_and_normalization():
    qp=torch.zeros(3,36);qp[:,3]=1
    qv=torch.randn(3,35)
    origin=torch.tensor([[0.,0.,0.],[10.,20.,0.],[30.,40.,0.]])
    data=SimpleNamespace(qpos=qp,qvel=qv,qacc_warmstart=torch.ones(3,35))
    calls=[]
    env=SimpleNamespace(sim=SimpleNamespace(data=data,forward=lambda:calls.append(1)),
                        scene=SimpleNamespace(env_origins=origin))
    io=StateIO(env)
    before=io.read()
    state=before[1:2].clone();state[:,3:7]=quat_from_rotvec(torch.tensor([[.3,.4,.8]]))*2
    io.write(torch.tensor([1]),state)
    expected=state.clone();expected[:,3:7]/=2
    torch.testing.assert_close(io.read()[1:2],expected)
    torch.testing.assert_close(io.read()[[0,2]],before[[0,2]])
    assert calls==[1] and not data.qacc_warmstart[1].any()

def test_old_checkpoint_rejected_before_loading_weights(tmp_path):
    import pytest
    from supermimic.training.loop import Trainer
    p=tmp_path/"old.pt";torch.save({"wm":{}},p)
    with pytest.raises(ValueError,match="old dev/FSQ"):
        Trainer.load(SimpleNamespace(device="cpu"),p)
