import torch
from supermimic.envs.mdp.rewards import tracking_reward_terms,TrackingRewardCfg
from supermimic.world_model.heads import AnalyticHeads,ReferenceFrame

def test_perfect_pose_max_reward_and_action_rate():
    p=torch.zeros(2,14,3);q=torch.zeros(2,14,4);q[...,0]=1
    a=torch.zeros(2,29)
    kwargs=dict(robot_body_pos_w=p,robot_body_quat_w=q,robot_body_lin_vel_w=p,
        robot_body_ang_vel_w=p,ref_body_pos_w=p,ref_body_quat_w=q,
        ref_body_lin_vel_w=p,ref_body_ang_vel_w=p,anchor_index=0,
        action=a,prev_action=a,joint_pos=a,soft_limits=torch.tensor([[-1.,1.]]).expand(29,-1),cfg=TrackingRewardCfg())
    r=tracking_reward_terms(**kwargs)
    torch.testing.assert_close(r["total"],torch.full((2,),.1))
    kwargs["action"]=torch.ones_like(a)
    torch.testing.assert_close(tracking_reward_terms(**kwargs)["action_rate_l2"],torch.full((2,),-.058))

def test_soft_termination_threshold():
    s=torch.zeros(1,71);s[:,3]=1
    ref=ReferenceFrame(None,None,None,None,torch.tensor([[0.,0.,.25]]),s[:,3:7])
    h=AnalyticHeads.__new__(AnalyticHeads)
    d=h.soft_termination(s,ref)
    torch.testing.assert_close(d,torch.full((1,),.5),atol=1e-6,rtol=1e-6)
    assert not h.terminated(s,ref).any()
    s[:,2]=-.01
    assert h.terminated(s,ref).all()
