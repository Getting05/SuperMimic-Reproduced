"""CPU reference storage must preserve GPU outputs, including clip-end goals."""
from dataclasses import astuple

import numpy as np
import pytest
import torch

from supermimic.envs.mdp.motion_library import MotionLibrary

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("compact", [False, True])
def test_reference_storage(tmp_path, compact):
    # Full mode needs no robot FK; compact mode exercises the real P2 FK.
    from supermimic.envs.p2 import P2_JOINT_NAMES

    files = []
    names = ("pelvis", "left_knee_link")
    for i in range(2):
        p = tmp_path / f"clip{i}.npz"
        pos = np.zeros((120, 2, 3), np.float32)
        pos[:, :, 2] = 0.85
        pos[:, :, 0] = np.arange(120)[:, None] * 0.001
        quat = np.zeros((120, 2, 4), np.float32)
        quat[:, :, 0] = 1
        np.savez(p, joint_names=P2_JOINT_NAMES, body_names=names,
                 body_pos_w=pos, body_quat_w=quat,
                 body_lin_vel_w=np.zeros_like(pos), body_ang_vel_w=np.zeros_like(pos),
                 joint_pos=np.zeros((120, 29), np.float32), joint_vel=np.zeros((120, 29), np.float32))
        files.append(p)
    kwargs = dict(npz_files=files, body_names=names, default_joint_pos=torch.zeros(29),
                  device="cuda:0", compact=compact, robot_variant="p2")
    gpu = MotionLibrary(**kwargs, library_device="gpu")
    cpu = MotionLibrary(**kwargs, library_device="cpu")
    assert gpu._ref.is_cuda and not cpu._ref.is_cuda
    frames = torch.tensor([0, 119, 120, 239], device="cuda:0")
    for x, y in [(gpu.ref71(frames), cpu.ref71(frames)),
                 (gpu.goal_window(frames, 10), cpu.goal_window(frames, 10))]:
        assert y.is_cuda
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    for x, y in zip(astuple(gpu.bodies(frames)), astuple(cpu.bodies(frames))):
        assert y.is_cuda
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    torch.testing.assert_close(cpu.last_frame_of(frames), torch.tensor([119, 119, 239, 239], device="cuda:0"))
