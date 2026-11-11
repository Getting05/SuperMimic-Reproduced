"""Convert ProtoMotions PyRoki P2 clips to SuperMimic motion files."""

from pathlib import Path

import tyro

from supermimic.data.motion_convert import convert_p2_retarget_to_npz


def main(source_dir: Path, output_dir: Path) -> None:
    sources = sorted(source_dir.rglob("*_retargeted.npz"))
    if not sources:
        raise FileNotFoundError(f"No retargeted clips under {source_dir}")
    if len({source.name for source in sources}) != len(sources):
        raise ValueError("Duplicate clip names. Convert each source subset into its own output directory.")
    for source in sources:
        convert_p2_retarget_to_npz(source, output_dir / source.name)


if __name__ == "__main__":
    tyro.cli(main)
