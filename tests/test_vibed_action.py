from types import SimpleNamespace
import torch
from supermimic.training.loop import execution_action
from supermimic.envs.actions import NoisyJointPositionAction

def test_execution_noise_once_and_clip():
    x=torch.zeros(10000,29)
    torch.manual_seed(5);expected=(x+.1*torch.randn_like(x)).clamp(-20,20)
    torch.manual_seed(5);torch.testing.assert_close(execution_action(x,.1),expected)
    torch.testing.assert_close(execution_action(x,0),x)
    assert execution_action(torch.full((2,29),100.),0).max()==20

def test_action_no_internal_noise_and_bias_sign():
    obj=NoisyJointPositionAction.__new__(NoisyJointPositionAction)
    obj._raw_actions=torch.zeros(2,29);obj._scale=2.;obj._offset=torch.ones(2,29)
    obj.noise_std=100.;obj.cfg=SimpleNamespace(clip=None)
    obj._target_ids=torch.arange(29)
    targets=[]
    obj._entity=SimpleNamespace(data=SimpleNamespace(encoder_bias=torch.full((2,29),.01)),
        set_joint_position_target=lambda target,**kw:targets.append(target))
    a=torch.full((2,29),.2)
    obj.process_actions(a);obj.apply_actions()
    torch.testing.assert_close(obj.executed_raw,a)
    torch.testing.assert_close(targets[0],torch.full((2,29),1.41))
