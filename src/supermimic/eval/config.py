"""Configuration shared by standalone and training-triggered evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

METRICS = {
    "pelvis_position": "m",
    "pelvis_orientation": "rad",
    "body_position_world": "m",
    "body_orientation_world": "rad",
    "body_position_reanchored": "m",
    "body_orientation_reanchored": "rad",
    "body_position_root_relative": "m",
    "joint_position": "rad",
    "joint_velocity": "rad/s",
}


@dataclass
class DatasetCfg:
    name: str = "val"
    path: Path = Path("data/npz/lafan1")
    format: Literal["auto", "npz", "packed"] = "auto"
    clips: list[str] = field(default_factory=list)
    clip_list: Path | None = None
    manifest: Path | None = None
    limit: int | None = None


@dataclass
class ResourceCfg:
    device: Literal["cpu", "cuda"] = "cpu"
    # Physical GPU IDs; each worker sees exactly one GPU as cuda:0.
    gpu_ids: list[int] = field(default_factory=lambda: [0])
    workers: int = 1
    threads_per_worker: int = 1
    num_envs: int = 16
    cpu_ids: list[int] = field(default_factory=list)
    cpu_count: int | None = None
    cpu_fraction: float | None = None
    library_device: Literal["cpu", "gpu"] = "cpu"


@dataclass
class QualityCfg:
    enabled: bool = False
    min_completion: float = 0.98
    # Maximum mean error during the executed trajectory, in the registry's SI units.
    max_errors: dict[str, float] = field(default_factory=lambda: {
        "pelvis_position": 0.30,
        "body_position_root_relative": 0.25,
        "joint_position": 0.6981317007977318,
    })


@dataclass
class EvalConfig:
    checkpoint: Path = Path("logs/run/ckpt_latest.pt")
    # Optional saved training config; must match the checkpoint's model contract.
    checkpoint_config: Path | None = None
    output_dir: Path = Path("evals/standalone")
    datasets: list[DatasetCfg] = field(default_factory=lambda: [DatasetCfg()])
    resources: ResourceCfg = field(default_factory=ResourceCfg)
    metrics: list[str] = field(default_factory=lambda: list(METRICS))
    reward_terms: bool = True
    quality: QualityCfg = field(default_factory=QualityCfg)
    seed: int = 0
    # Fixed actuator delay used by the earlier heldout CPU evaluations.
    actuator_delay: int = 2

    def validate(self):
        r = self.resources
        if not self.datasets or len({d.name for d in self.datasets}) != len(self.datasets):
            raise ValueError("Provide datasets with unique names")
        if min(r.workers, r.threads_per_worker, r.num_envs) < 1:
            raise ValueError("workers, threads_per_worker and num_envs must be positive")
        if r.device not in ("cpu", "cuda") or r.library_device not in ("cpu", "gpu"):
            raise ValueError("Invalid evaluation device")
        if r.device == "cuda" and (not r.gpu_ids or len(set(r.gpu_ids)) != len(r.gpu_ids)):
            raise ValueError("Provide unique GPU IDs")
        if any(i < 0 for i in r.gpu_ids + r.cpu_ids) or len(set(r.cpu_ids)) != len(r.cpu_ids):
            raise ValueError("Resource IDs must be nonnegative and unique")
        if sum([bool(r.cpu_ids), r.cpu_count is not None, r.cpu_fraction is not None]) > 1:
            raise ValueError("Choose cpu_ids, cpu_count or cpu_fraction")
        if r.cpu_count is not None and r.cpu_count < 1:
            raise ValueError("cpu_count must be positive")
        if r.cpu_fraction is not None and not 0 < r.cpu_fraction <= 1:
            raise ValueError("cpu_fraction must be in (0, 1]")
        names = set(self.metrics) | (set(self.quality.max_errors) if self.quality.enabled else set())
        if names - METRICS.keys():
            raise ValueError(f"Unknown metrics: {sorted(names - METRICS.keys())}")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("Duplicate metrics")
        if not 0 <= self.quality.min_completion <= 1 or any(v < 0 for v in self.quality.max_errors.values()):
            raise ValueError("Invalid quality thresholds")
        if self.actuator_delay < 0:
            raise ValueError("actuator_delay must be nonnegative")
        for d in self.datasets:
            if d.limit is not None and d.limit < 1:
                raise ValueError("Dataset limit must be positive")
            if d.format not in ("auto", "packed", "npz"):
                raise ValueError("Invalid dataset format")


@dataclass
class AutoEvalCfg:
    enabled: bool = False
    interval: int = 4000
    on_start: bool = False
    on_finish: bool = True
    wait_on_finish: bool = True
    # Backpressure: skip a trigger while evaluation is busy, avoiding an
    # unbounded queue and keeping the configured CPU/GPU budget fixed.
    # Override settings from a standalone evaluation config, if provided.
    config_file: Path | None = None
    settings: EvalConfig = field(default_factory=EvalConfig)
