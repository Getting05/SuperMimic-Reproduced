import numpy as np
import torch
from supermimic.envs.p2 import load_p2_model
from supermimic.envs.state_io import from_native
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg


def test_critic_reuses_rollout_fk_with_matching_gradients():
    model=load_p2_model();fk=G1Kinematics(model)
    names=fk.body_names[:14]
    heads=AnalyticHeads(fk,names,'pelvis','pelvis',
        torch.tensor(model.jnt_range[1:],dtype=torch.float32),torch.tensor(model.qpos0[7:],dtype=torch.float32))
    qpos=torch.tensor(np.tile(model.qpos0,(4,1)),dtype=torch.float32)
    qvel=torch.randn(4,35)*.01
    state=from_native(qpos,qvel);xi=torch.zeros(4,33)
    body=heads.robot_bodies(state)
    ref=ReferenceFrame(body['pos']+.01,body['quat'],body['lin_vel'],body['ang_vel'],state[:,:3]+.01,state[:,3:7])
    spec=StateSpec(StateSpecCfg(),'cpu');result=[]
    for cached in [False,True]:
        s=state.clone().requires_grad_()
        pos=heads.robot_bodies(s)['pos'] if cached else None
        features=heads.critic_features(s,ref,xi,spec,body_pos=pos)
        grad=torch.autograd.grad(features.square().mean(),s)[0]
        result.append((features,grad))
    for a,b in zip(*result):torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)


def test_geometry_heads_capture_as_one_graph_on_cpu():
    from supermimic.world_model.world_model import WorldModel,WorldModelCfg
    from supermimic.imagination.compiled import ImagStep
    model=load_p2_model();fk=G1Kinematics(model)
    qpos=torch.tensor(np.tile(model.qpos0,(2,1)),dtype=torch.float32)
    s=from_native(qpos,torch.zeros(2,35));xi=torch.zeros(2,33)
    q0=torch.tensor(model.qpos0[7:],dtype=torch.float32)
    heads=AnalyticHeads(fk,fk.body_names[:14],'pelvis','pelvis',torch.tensor(model.jnt_range[1:],dtype=torch.float32),q0)
    body=heads.robot_bodies(s)
    ref=ReferenceFrame(body['pos'],body['quat'],body['lin_vel'],body['ang_vel'],s[:,:3],s[:,3:7])
    wm=WorldModel(WorldModelCfg(hidden=(32,32,32)),heads,'cpu');wm.requires_grad_(False)
    step=ImagStep(wm,q0)
    args=(s,torch.zeros(2,29),torch.zeros(2,29),xi,ref.body_pos_w,ref.body_quat_w,ref.body_lin_vel_w,ref.body_ang_vel_w,ref.root_pos_w,ref.root_quat_w)
    eager=step._step(*args)
    # fullgraph=True rejects graph breaks rather than silently running fragmented eager code.
    s_next=wm.step(s,args[1],xi)
    captured=torch.compile(step._heads,backend='eager',fullgraph=True)(s_next,*args[1:])
    expected=(eager[0],eager[1],eager[2],eager[3],eager[5],eager[6])
    for a,b in zip(expected,captured):torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
