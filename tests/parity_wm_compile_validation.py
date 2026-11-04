"""Default-width, 2048x8 eager/compiled WM forward/backward validation."""
import json,time
from pathlib import Path
import torch
from supermimic.world_model.world_model import WorldModel,WorldModelCfg
torch.manual_seed(71)
dev='cuda:0';b=2048;k=8
wm=WorldModel(WorldModelCfg(),None,dev)
s=torch.zeros(k+1,b,71,device=dev);s[:,:,3]=1;s[:,:,2]=.75
s[:,:,13:42]=torch.randn(k+1,b,29,device=dev)*.01
a=torch.randn(k,b,29,device=dev)*.1;xi=torch.zeros(b,33,device=dev)
wm.update_normalization(s[:-1].flatten(0,1),s[1:].flatten(0,1),a.flatten(0,1),
                        xi[None].expand(k,-1,-1).flatten(0,1))
mask=torch.ones(k,b,dtype=torch.bool,device=dev)
result={}
for name,mode in [('eager',None),('compiled','default')]:
 wm.zero_grad(set_to_none=True);wm.enable_compile(mode)
 start=time.perf_counter()
 loss,m=wm.window_loss(s,a,xi,torch.zeros(k,b,dtype=torch.bool,device=dev),None,mask)
 loss.backward();torch.cuda.synchronize()
 norm=torch.nn.utils.clip_grad_norm_(wm.parameters(),1.)
 assert torch.isfinite(loss) and torch.isfinite(norm)
 result[name]=dict(loss=float(loss.detach()),gradient_norm=float(norm),seconds=time.perf_counter()-start)
 print(name,result[name],flush=True)
assert abs(result['eager']['loss']-result['compiled']['loss'])<1e-4
result['peak_gib']=torch.cuda.max_memory_allocated()/1024**3
p=Path('logs/validation_20261104/wm_compile.json')
p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2))
print('DEFAULT_WM_COMPILE_BACKWARD_OK',flush=True)
