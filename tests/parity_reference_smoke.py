"""CPU numerical comparison against the pinned, separately checked-out Vibed implementation."""
import sys,types,os
from pathlib import Path
import torch
root=Path(os.environ.get('VIBED_REFERENCE_ROOT','/root/code/vibed-parity-reference'))/'supermimic'
pkg=types.ModuleType('vibed_reference');pkg.__path__=[str(root)];sys.modules[pkg.__name__]=pkg
# Pure tensor comparisons do not instantiate the reference FK or Isaac scene.
# Supply only its annotation name to avoid importing robot.py's Isaac dependency.
kin=types.ModuleType('vibed_reference.kinematics');kin.G1Kinematics=object
sys.modules[kin.__name__]=kin
from vibed_reference.networks import WorldModel as VWM,RunningNorm as VNorm
from vibed_reference.mdp import invariant_features,unflatten_state
from vibed_reference.motion import MotionLib
from supermimic.policy.normalization import RunningNorm
from supermimic.world_model.world_model import WorldModel,WorldModelCfg
from supermimic.world_model.state_spec import StateSpec,StateSpecCfg
from supermimic.kinematics.rotations import quat_from_rotvec,quat_apply_inverse,matrix_from_quat
from supermimic.envs.mdp.motion_library import MotionLibrary
torch.manual_seed(71)
s=torch.randn(3,71);s[:,3:7]=quat_from_rotvec(torch.randn(3,3)*.4)
s1=s.clone();s1[:,0:3]+=.01;s1[:,3:7]=quat_from_rotvec(torch.randn(3,3)*.4)
spec=StateSpec(StateSpecCfg(),'cpu')
torch.testing.assert_close(spec.canonical(s),invariant_features(unflatten_state(s)),atol=2e-6,rtol=1e-5)
torch.testing.assert_close(spec.targets(s,s1),VWM.targets(unflatten_state(s),unflatten_state(s1)),atol=2e-6,rtol=1e-5)
ours=WorldModel(WorldModelCfg(hidden=(32,32,32)),None,'cpu')
theirs=VWM(hidden=(32,32,32))
delta=torch.randn(70)*.01
ours.out_norm.mean.copy_(delta);theirs.out_norm.mean.copy_(delta)
a=torch.zeros(3,29);xi=torch.zeros(3,33)
from vibed_reference.mdp import flatten_state
torch.testing.assert_close(ours.step(s,a,xi),flatten_state(theirs.step(unflatten_state(s),a,torch.zeros_like(xi))),atol=3e-6,rtol=2e-5)
n=RunningNorm(7);v=VNorm(7);x=torch.randn(32,7)
for chunk in (x[:11],x[11:]):n.update(chunk);v.update(chunk)
torch.testing.assert_close(n(x),v(x),atol=1e-6,rtol=1e-5)
lib=MotionLibrary.__new__(MotionLibrary)
lib.device=lib.storage_device=torch.device('cpu');lib.dtype=torch.float32;lib._ref=s
lib.frame_clip=torch.zeros(3,dtype=torch.long);lib.clip_last=torch.tensor([2])
m=MotionLib.__new__(MotionLib)
m.clip_id=lib.frame_clip;m.clip_end=lib.clip_last
m.root_p=s[:,:3];m.root_R=matrix_from_quat(s[:,3:7])
m.root_v_local=quat_apply_inverse(s[:,3:7],s[:,7:10])
m.root_w_local=quat_apply_inverse(s[:,3:7],s[:,10:13])
m.q=s[:,13:42];m.qd=s[:,42:71]
torch.testing.assert_close(lib.goal_window(torch.tensor([0,1]),10),m.goal(torch.tensor([0,1]),10),atol=2e-6,rtol=1e-5)
print('PINNED_VIBED_NUMERICS_OK: invariant71 targets70 integration71 norm goal730')
