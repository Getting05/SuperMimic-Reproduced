"""Compatibility wrapper around the unified evaluation engine."""
from supermimic.eval.legacy import packed_main

if __name__ == "__main__":
    packed_main("cuda")
