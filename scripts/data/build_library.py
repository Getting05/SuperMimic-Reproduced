"""Pack filtered npz clips into one compact library file (fp16 71-d reference states).

  uv run python scripts/data/build_library.py --npz-dir data/npz/bones_seed --manifest data/npz/bones_seed/manifest_filtered.csv --out data/library/bones_seed_compact.pt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import tyro
from tqdm import tqdm

from supermimic.envs.mdp.motion_library import discover_npz


def main(npz_dir: Path, out: Path, manifest: Path | None = None, workers: int = 16) -> None:
    files = discover_npz(npz_dir, manifest=manifest)
    print(f"{len(files)} clips")
    from concurrent.futures import ThreadPoolExecutor

    def load(f):
        with np.load(f) as d:
            names = [str(n) for n in d["body_names"]]
            r = names.index("pelvis")
            ref = np.concatenate([d["body_pos_w"][:, r], d["body_quat_w"][:, r], d["body_lin_vel_w"][:, r], d["body_ang_vel_w"][:, r], d["joint_pos"], d["joint_vel"]], axis=1)
            joints = tuple(d.get("joint_names", ()))
        return f.stem, ref.astype(np.float16), joints

    with ThreadPoolExecutor(workers) as ex:
        items = list(tqdm(ex.map(load, files), total=len(files)))
    names = [n for n, _, _ in items]
    lengths = [int(r.shape[0]) for _, r, _ in items]
    starts = np.cumsum([0] + lengths[:-1]).tolist()
    ref = torch.from_numpy(np.concatenate([r for _, r, _ in items], 0))
    joint_names = items[0][2]
    if any(joints != joint_names for _, _, joints in items):
        raise ValueError("All clips must have the same joint order")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_layout": "vibed71", "ref": ref, "names": names, "starts": starts, "lengths": lengths, "joint_names": joint_names}, out)
    print(f"saved {out}: {ref.shape[0]} frames ({ref.shape[0]/50/3600:.2f} h), {ref.numel()*2/1e9:.2f} GB fp16")


if __name__ == "__main__":
    tyro.cli(main)
