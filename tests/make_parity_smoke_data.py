"""Create a reproducible synthetic P2 clip for pipeline tests."""
import sys
from pathlib import Path
import numpy as np
from supermimic.envs.p2 import P2_JOINT_NAMES
from supermimic.data.motion_convert import convert_p2_retarget_to_npz
out=Path(sys.argv[1] if len(sys.argv)>1 else "/tmp/parity-smoke-data")
out.mkdir(parents=True,exist_ok=True)
source=out/"source.npz"
np.savez(source,joint_names=P2_JOINT_NAMES,joint_angles=np.zeros((100,29)),
         base_frame_pos=np.tile([0.,0.,.75],(100,1)),
         base_frame_wxyz=np.tile([1.,0.,0.,0.],(100,1)),fps=50)
convert_p2_retarget_to_npz(source,out/"motions/stand.npz")
print(out/"motions")
