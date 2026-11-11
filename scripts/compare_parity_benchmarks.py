"""Reaggregate original dev/FSQ evaluation with the new full-clip denominator."""
import json
from pathlib import Path
import tyro

def compare(baseline, candidate, baseline_sim_transitions, candidate_sim_transitions):
    if baseline_sim_transitions!=candidate_sim_transitions:
        raise ValueError("A/B requires equal actual simulator transition budgets")
    old={r["name"]:r for r in baseline["clips"]}
    new={r["name"]:r for r in candidate["clips"]}
    if old.keys()!=new.keys():raise ValueError("A/B requires identical clip sets")
    rows=[]
    for name,a in old.items():
        b=new[name]
        if a["frames"]!=b["frames"]:raise ValueError(f"Clip length mismatch: {name}")
        full=a["norm_return"]*a["steps"]/max(a["frames"]-1,1)
        rows.append(dict(name=name,frames=a["frames"],baseline=full,candidate=b["norm_return"],
                         delta=b["norm_return"]-full))
    return dict(sim_transitions=baseline_sim_transitions,clips=rows,
                mean_delta=sum(r["delta"] for r in rows)/max(len(rows),1))

def main(baseline_json:Path,candidate_json:Path,baseline_sim_transitions:int,
         candidate_sim_transitions:int,out:Path):
    result=compare(json.loads(baseline_json.read_text()),json.loads(candidate_json.read_text()),
                   baseline_sim_transitions,candidate_sim_transitions)
    out.write_text(json.dumps(result,indent=2))
    print(result["mean_delta"])

if __name__=="__main__":tyro.cli(main)
