from types import SimpleNamespace
import torch
from supermimic.envs.mdp import rewards
from supermimic.envs import state_io


def test_reward_pass_refreshes_even_in_same_step(monkeypatch):
    calls=[]
    def reward(state,*args):
        calls.append(state.clone())
        return {name:state+index for index,name in enumerate(['a','b','c'])}
    env=SimpleNamespace(common_step_counter=7,state=torch.tensor([1.,2.]),
        action_manager=SimpleNamespace(action=None,prev_action=None),step_dt=.02)
    monkeypatch.setattr(state_io,'StateIO',lambda e:SimpleNamespace(read=lambda:e.state))
    monkeypatch.setattr(rewards,'simulator_heads',lambda e:SimpleNamespace(reward=reward))
    monkeypatch.setattr(rewards,'next_reference',lambda e:None)
    for state in [torch.tensor([1.,2.]),torch.tensor([4.,9.])]:
        # A second pass or partial reset can occur without changing the step counter.
        env.state=state
        for i,name in enumerate(['a','b','c']):
            actual=rewards.simulator_reward_term(env,name,1.,reuse=i>0)
            torch.testing.assert_close(actual,(state+i)/.02,atol=0,rtol=0)
    assert len(calls)==2
    env.common_step_counter+=1
    rewards.simulator_reward_term(env,'b',1.,reuse=True)
    assert len(calls)==3


def test_termination_reference_skips_body_fk():
    from supermimic.world_model.heads import ReferenceFrame
    p=torch.randn(3,3);q=torch.zeros(3,4);q[:,0]=1
    lib=SimpleNamespace(last_frame_of=lambda frame:torch.full_like(frame,10),
        ref=lambda frame:SimpleNamespace(root_pos_w=p,root_quat_w=q))
    env=SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name:
        SimpleNamespace(frame=torch.tensor([0,5,10]),library=lib)))
    # The root-only path must not request lib.bodies (which is deliberately absent).
    ref=rewards.next_reference(env,bodies=False)
    assert isinstance(ref,ReferenceFrame)
    assert ref.body_pos_w is None
    torch.testing.assert_close(ref.root_pos_w,p,atol=0,rtol=0)
    torch.testing.assert_close(ref.root_quat_w,q,atol=0,rtol=0)
