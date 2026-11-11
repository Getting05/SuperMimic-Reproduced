"""Validate multi-rank SGS library layout before using rank-local sampler bins."""

from pathlib import Path


def validate_sgs_rank_libraries(
    signatures: list[str | None],
    library_files: list[str | None],
    rank_sharded: bool,
) -> bool:
    """Return whether SGS uses rank-local data and statistics.

    Shared datasets retain the original behavior. If different datasets are
    loaded, rank sharding must be explicit and each rank must own a distinct,
    correctly named library. A mix of repeated and distinct datasets is rejected
    because global bin indexes cannot be assumed to align between ranks.
    """
    if not signatures or len(signatures) != len(library_files) or any(s is None for s in signatures):
        raise ValueError("SGS requires a valid MotionLibrary signature for every rank")
    if len(set(signatures)) == 1:
        return False

    expected = [f"rank_{rank:02d}.pt" for rank in range(len(signatures))]
    if not rank_sharded:
        raise ValueError(
            "SGS ranks must share the same MotionLibrary or enable "
            "--env.sgs-rank-sharded with distinct rank_XX.pt files"
        )
    if [Path(p).name if p else None for p in library_files] != expected:
        raise ValueError(f"Rank-sharded SGS requires per-rank library files {expected}")
    if len(set(signatures)) != len(signatures):
        raise ValueError("Rank-sharded SGS requires a different MotionLibrary for every rank")
    return True
