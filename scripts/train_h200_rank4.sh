#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 0. Environment
# ============================================================

PROJECT="${PROJECT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"

# 必须在 source 激活脚本之前设置 GPU 环境变量
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,5}"
export SM_DIST_BACKEND=nccl

cd "$PROJECT"

# 某些自定义激活脚本会引用未定义变量
# 暂时关闭 nounset，激活后重新开启
set +u
source "$PROJECT/.venv/bin/activate-docker150"
set -u

export PYTHONUNBUFFERED=1

echo "============================================================"
echo "SuperMimic H200 4-GPU Training"
echo "============================================================"
echo "Project: $PROJECT"
echo "Python:  $(which python)"
echo "GPU:     $CUDA_VISIBLE_DEVICES"
echo "============================================================"

# ============================================================
# 1. Configuration
# ============================================================

DATASET="$PROJECT/data/motions/mix5clean_v2_supermimic_p2_50hz"
CACHE="$DATASET/train_h200_rank4"

LOG_DIR="$PROJECT/logs"
mkdir -p "$LOG_DIR"

RUN_NAME="${RUN_NAME:-p2_fsq_mix5clean_h200_4gpu_$(date +%Y%m%d_%H%M%S)}"

# Per rank: best balanced training-sample throughput measured with 8192 envs.
# Keep WM:policy batch ratio 1:2; overrides are available for new comparisons.
# Benchmark results: logs/h200_fixed8192_batch_bench_20261103/results.json
NUM_ENVS="${NUM_ENVS:-4096}"
WM_BATCH_SIZE="${WM_BATCH_SIZE:-2048}"
POLICY_BATCH_SIZE="${POLICY_BATCH_SIZE:-4096}"
LIBRARY_DEVICE="${LIBRARY_DEVICE:-gpu}"
# Excluded from training, available for concurrent evaluation.
EVAL_GPU="${EVAL_GPU:-0}"
TOTAL_ITERS="${TOTAL_ITERS:-320000}"

# ============================================================
# 2. GPU validation
# ============================================================

if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^[0-7](,[0-7]){3}$ ]]; then
    echo "ERROR: 必须指定恰好 4 张 GPU，例如 1,2,3,5" >&2
    exit 1
fi

IFS=, read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"

if [[ $(printf '%s\n' "${gpu_ids[@]}" | sort -u | wc -l) -ne 4 ]]; then
    echo "ERROR: GPU 编号不能重复" >&2
    exit 1
fi

if [[ "$LIBRARY_DEVICE" != gpu && "$LIBRARY_DEVICE" != cpu ]]; then
    echo "ERROR: LIBRARY_DEVICE must be gpu or cpu" >&2
    exit 1
fi
for gpu in "${gpu_ids[@]}"; do
    if [[ "$gpu" == "$EVAL_GPU" ]]; then
        echo "ERROR: EVAL_GPU=$EVAL_GPU is also selected for training" >&2
        exit 1
    fi
done

check_gpus() {
    local gpu busy

    for gpu in "${gpu_ids[@]}"; do
        busy=$(nvidia-smi -i "$gpu" \
            --query-compute-apps=pid,used_memory \
            --format=csv,noheader)

        if [[ -n "$busy" ]]; then
            printf 'ERROR: GPU %s 已有计算进程：\n%s\n' \
                "$gpu" "$busy" >&2
            return 1
        fi
    done
}

echo ""
echo "[1/4] Checking GPUs..."
nvidia-smi -L
check_gpus

# 检查 PyTorch 能否识别四张卡
python - <<'PY'
import torch

count = torch.cuda.device_count()
print(f"PyTorch detected {count} GPUs")

if count != 4:
    raise RuntimeError(f"Expected 4 GPUs, got {count}")

for i in range(count):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
PY

# ============================================================
# 3. Build rank-specific training caches
# ============================================================

echo ""
echo "[2/4] Checking / building rank caches..."

flock "$DATASET/.train_h200_rank4.lock" \
python - "$DATASET" "$CACHE" <<'PY'

import json
import os
import shutil
import sys
import tempfile

from pathlib import Path

import torch

dataset, cache = map(Path, sys.argv[1:])

manifest = json.loads(
    (dataset / "COMPLETE.json").read_text()
)

train = manifest["splits"]["train"]

shards = sorted(
    train["shards"],
    key=lambda entry: entry["file"]
)

assert len(shards) == 32, (
    f"Expected 32 shards, got {len(shards)}"
)

# 每个 rank 分配 8 个 shard
groups = [
    shards[rank::4]
    for rank in range(4)
]

plan = [
    [entry["sha256"] for entry in group]
    for group in groups
]

# ------------------------------------------------------------
# Cache validation
# ------------------------------------------------------------

if cache.exists():

    receipt = cache / "COMPLETE.json"

    if (
        not receipt.is_file()
        or json.loads(receipt.read_text()) != plan
    ):
        raise RuntimeError(
            f"Rank cache incomplete or mismatched: {cache}"
        )

    for rank in range(4):
        path = cache / f"rank_{rank:02d}.pt"

        if not path.is_file():
            raise FileNotFoundError(path)

    print(f"Reusing existing cache: {cache}", flush=True)

    sys.exit(0)

# ------------------------------------------------------------
# Build cache atomically
# ------------------------------------------------------------

stage = Path(
    tempfile.mkdtemp(
        prefix=".train_h200_rank4-",
        dir=dataset
    )
)

try:

    joint_names = manifest["joint_names"]

    total_clips = 0
    total_frames = 0

    for rank, group in enumerate(groups):

        expected_frames = sum(
            entry["frames"]
            for entry in group
        )

        ref = torch.empty(
            (expected_frames, 71),
            dtype=torch.float16
        )

        names = []
        starts = []
        lengths = []

        offset = 0
        state_layout = None

        for entry in group:

            path = dataset / entry["file"]

            assert path.stat().st_size == entry["bytes"], path

            pack = torch.load(
                path,
                map_location="cpu",
                mmap=True,
                weights_only=True
            )

            layout=pack.get("state_layout","p_quat_q_v_w_world_qd")
            if state_layout is not None and layout != state_layout:
                raise ValueError("Mixed state layouts in rank cache")
            state_layout=layout
            frames = entry["frames"]

            assert list(pack["joint_names"]) == joint_names
            assert pack["ref"].shape == (frames, 71)
            assert pack["ref"].dtype == torch.float16

            assert (
                len(pack["names"])
                == len(pack["starts"])
                == len(pack["lengths"])
                == entry["clips"]
            )

            ref[offset:offset + frames].copy_(
                pack["ref"]
            )

            names.extend(pack["names"])

            starts.extend(
                offset + int(start)
                for start in pack["starts"]
            )

            lengths.extend(
                int(length)
                for length in pack["lengths"]
            )

            offset += frames

            del pack

        output = stage / f"rank_{rank:02d}.pt"

        torch.save(
            dict(
                state_layout=state_layout,
                ref=ref,
                names=names,
                starts=starts,
                lengths=lengths,
                joint_names=joint_names
            ),
            output
        )

        total_clips += len(names)
        total_frames += offset

        print(
            f"Rank {rank}: "
            f"{len(names):,} clips, "
            f"{offset:,} frames, "
            f"{output.stat().st_size / 1024**3:.2f} GB",
            flush=True
        )

        del ref

    assert total_clips == train["clips"], (
        total_clips, train["clips"]
    )

    assert total_frames == train["frames"], (
        total_frames, train["frames"]
    )

    (stage / "COMPLETE.json").write_text(
        json.dumps(plan) + "\n"
    )

    os.rename(stage, cache)

    print(
        f"Successfully built rank caches: {cache}",
        flush=True
    )

finally:

    if stage.exists():
        shutil.rmtree(stage)

PY

# ============================================================
# 4. Recheck GPUs before training
# ============================================================

echo ""
echo "[3/4] Rechecking GPUs..."

check_gpus

echo ""
echo "GPU status:"
nvidia-smi

# ============================================================
# 5. Distributed training
# ============================================================

echo ""
echo "[4/4] Starting distributed training..."

echo "Run name:    $RUN_NAME"
echo "Num envs:    $NUM_ENVS"
echo "Iterations:  $TOTAL_ITERS"
echo "GPUs:        $CUDA_VISIBLE_DEVICES"
echo "Cache:       $CACHE"
echo "Library:     $LIBRARY_DEVICE (reference storage)"
echo "WM batch:    $WM_BATCH_SIZE per rank"
echo "Policy batch:$POLICY_BATCH_SIZE per rank"
echo "Eval GPU:    $EVAL_GPU reserved by this launcher"
echo "W&B:         enabled"
echo ""

# torchrun creates 4 workers.
# Each worker loads its own rank-specific dataset.

exec torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=4 \
    --no-python \
    bash -c '
        set -euo pipefail

        cache=$1
        shift

        printf -v rank_file "%s/rank_%02d.pt" \
            "$cache" "$LOCAL_RANK"

        echo "Rank $LOCAL_RANK loading: $rank_file"

        exec python -m supermimic.scripts.train \
            --data.library "$rank_file" "$@"

    ' _ "$CACHE" \
    --env.robot p2 \
    --policy.fsq \
    --env.num-envs "$NUM_ENVS" \
    --data.library-device "$LIBRARY_DEVICE" \
    --wm-trainer.batch-size "$WM_BATCH_SIZE" \
    --policy-trainer.batch-size "$POLICY_BATCH_SIZE" \
    --schedule.total-iterations "$TOTAL_ITERS" \
    --run-name "$RUN_NAME" \
    --wandb.enabled \
    --wandb.project fsq-supermimic \
    "$@"
