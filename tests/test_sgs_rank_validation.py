"""CPU-only regressions for rank-sharded SGS launch validation."""

import pytest

from supermimic.config import TrainConfig
from supermimic.training.sgs_rank_validation import validate_sgs_rank_libraries


def test_default_is_shared_sgs():
    cfg = TrainConfig()
    assert cfg.env.sgs_rank_sharded is False
    assert validate_sgs_rank_libraries(["same", "same"], ["motion.pt", "motion.pt"], False) is False


@pytest.mark.parametrize("enabled", [False, True])
def test_shared_library_backward_compatible(enabled):
    assert not validate_sgs_rank_libraries(["same", "same"], ["motion.pt", "motion.pt"], enabled)


def test_explicit_rank_shards_keep_local_statistics():
    assert validate_sgs_rank_libraries(
        ["sha0", "sha1", "sha2"], ["rank_00.pt", "rank_01.pt", "rank_02.pt"], True
    )


@pytest.mark.parametrize(
    "sigs,files,enabled",
    [
        (["sha0", "sha1"], ["rank_00.pt", "rank_01.pt"], False),
        (["sha0", "sha1"], ["rank_00.pt", "shared.pt"], True),
        (["sha0", "sha1"], ["rank_01.pt", "rank_00.pt"], True),
        (["sha0", "sha0", "sha1"], ["rank_00.pt", "rank_01.pt", "rank_02.pt"], True),
        ([None, "sha1"], ["rank_00.pt", "rank_01.pt"], True),
    ],
)
def test_reject_invalid_rank_sharding(sigs, files, enabled):
    with pytest.raises(ValueError):
        validate_sgs_rank_libraries(sigs, files, enabled)
