"""Explicit small-batch eager/Inductor numerical and timing comparison."""
import os,time
from pathlib import Path
import torch
from supermimic.config import TrainConfig
from supermimic.training.loop import Trainer
from supermimic.imagination.compiled import ImagStep
from supermimic.world_model.wm_trainer import reference_at
c=TrainConfig();c.env.robot='p2';c.env.num_envs=2;c.env.replay_capacity=32
c.data.npz_dir=Path(os.environ['PARITY_SMOKE_MOTIONS'])
c.log_root=Path('/tmp/parity-compile');c.world_model.hidden=(32,32,32)
c.policy.encoder_hidden=(32,);c.policy.decoder_hidden=(32,)
c.wm_trainer.compile=None;c.policy_trainer.compile=None
t=Trainer(c);t.env.reset();s=t.recorder.state_io.read()
a=torch.zeros(2,29,device=s.device);xi=t.recorder.ctx.read()
ref=reference_at(t.lib,torch.minimum(t.command.frame+1,t.lib.last_frame_of(t.command.frame)))
eager=ImagStep(t.wm,t.lib.default_joint_pos)
compiled=ImagStep(t.wm,t.lib.default_joint_pos,compile_mode='default')
with torch.no_grad():
 before=eager(s,a,a,xi,ref)
 start=time.perf_counter();after=compiled(s,a,a,xi,ref);torch.cuda.synchronize()
 print('compile_seconds',time.perf_counter()-start,flush=True)
 for x,y in zip(before,after):torch.testing.assert_close(x,y,atol=2e-4,rtol=2e-3)
 for label,fn in [('eager',eager),('compiled',compiled)]:
  start=time.perf_counter()
  for _ in range(8):fn(s,a,a,xi,ref)
  torch.cuda.synchronize();print(label,'8_steps_seconds',time.perf_counter()-start,flush=True)
print('COMPILE_NUMERICS_OK',flush=True);t.env.close()
