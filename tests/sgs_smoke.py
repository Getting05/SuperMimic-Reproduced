"""SGS entry point backed by the current parity training/resume smoke."""
import os
import runpy
from pathlib import Path
os.environ["PARITY_SMOKE_SAMPLER"]="sgs"
os.environ.setdefault("PARITY_SMOKE_UPDATES","2")
for old,new in (("SGS_SMOKE_LIBRARY","PARITY_SMOKE_LIBRARY"),
                ("SGS_SMOKE_OUTPUT","PARITY_SMOKE_OUTPUT")):
    if old in os.environ:os.environ[new]=os.environ[old]
if os.environ.get("SGS_SMOKE_PHASE")=="resume":os.environ["PARITY_SMOKE_RESUME"]="1"
runpy.run_path(str(Path(__file__).with_name("parity_training_smoke.py")),run_name="__main__")
