"""Rank-zero supervision of training-triggered evaluation subprocesses."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from supermimic.config_io import load_config
from supermimic.eval.config import EvalConfig
from supermimic.eval.runner import write_json


class AutoEvaluator:
    def __init__(self, config, log_dir):
        self.config = config
        self.log_dir = Path(log_dir)
        self.process = None
        self.current = None
        self.last_checkpoint = None
        self.history = []
        self.settings = load_config(config.config_file, EvalConfig()) if config.config_file else deepcopy(config.settings)
        self.settings.validate()

    def poll(self):
        if self.process is not None and self.process.poll() is not None:
            self.current["state"] = "completed" if self.process.returncode == 0 else "failed"
            self.current["exit_code"] = self.process.returncode
            print(f"[eval] {self.current['state']}: {self.current['output_dir']}", flush=True)
            self.process = None
            self._write_status()

    def submit(self, checkpoint):
        self.poll()
        checkpoint = Path(checkpoint).resolve()
        if checkpoint == self.last_checkpoint:
            return
        if self.process is not None:
            print(f"[eval] busy; skipped trigger for {checkpoint.name}", flush=True)
            self.history.append(dict(checkpoint=str(checkpoint), state="skipped_busy"))
            self._write_status()
            return
        cfg = deepcopy(self.settings)
        cfg.checkpoint = checkpoint
        cfg.output_dir = self.log_dir / "eval" / checkpoint.stem
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        path = cfg.output_dir / "resolved_config.json"
        write_json(path, asdict(cfg))
        env = dict(os.environ)
        # Controller children pick their physical GPU IDs from the evaluation
        # config, independent of the training rank's pinned GPU.
        original = env.get("SM_EVAL_VISIBLE_DEVICES")
        if original is not None:
            env["CUDA_VISIBLE_DEVICES"] = original
        with (cfg.output_dir / "controller.log").open("w") as log:
            self.process = subprocess.Popen([sys.executable, "-m", "supermimic.eval.runner", "--config", str(path)],
                                            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.last_checkpoint = checkpoint
        self.current = dict(checkpoint=str(checkpoint), output_dir=str(cfg.output_dir), state="running", pid=self.process.pid)
        self.history.append(self.current)
        self._write_status()
        print(f"[eval] started: {checkpoint.name} -> {cfg.output_dir}", flush=True)

    def _write_status(self):
        write_json(self.log_dir / "eval" / "automatic_status.json", self.history)

    def finish(self):
        if self.config.wait_on_finish:
            while self.process is not None:
                self.poll()
                if self.process is not None:
                    time.sleep(0.2)
        else:
            self.poll()
