"""Strict TOML/JSON dataclass configuration loading (CLI flags may override defaults)."""
from __future__ import annotations

import json
import types
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Union, get_args, get_origin, get_type_hints


def read_config(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text())
    import tomllib
    with path.open("rb") as stream:
        return tomllib.load(stream)


def update_config(instance, values: dict):
    """Apply a partial config, rejecting unknown keys and invalid scalar types."""
    hints = get_type_hints(type(instance))
    names = {f.name for f in fields(instance)}
    for key, value in values.items():
        if key not in names:
            raise ValueError(f"Unknown {type(instance).__name__} option: {key}")
        current = getattr(instance, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be a table")
            update_config(current, value)
        else:
            setattr(instance, key, _convert(hints[key], value))
    return instance


def _convert(annotation, value):
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (types.UnionType, Union):
        if value is None and type(None) in args:
            return None
        for option in args:
            if option is type(None):
                continue
            try:
                return _convert(option, value)
            except (TypeError, ValueError):
                pass
        raise ValueError(f"Invalid value {value!r} for {annotation}")
    if annotation is Path:
        return Path(value)
    if is_dataclass(annotation):
        return update_config(annotation(), value)
    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise ValueError("Expected an array")
        return origin(_convert(args[0], item) for item in value)
    if origin is dict:
        return {_convert(args[0], k): _convert(args[1], v) for k, v in value.items()}
    # Literal options are checked against their declared choices.
    from typing import Literal
    if origin is Literal:
        if value not in args:
            raise ValueError(f"Expected one of {args}, got {value!r}")
        return value
    if annotation is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if annotation in (str, int, float, bool) and type(value) is not annotation:
        raise ValueError(f"Expected {annotation.__name__}, got {value!r}")
    return value


def load_config(path, instance):
    return update_config(instance, read_config(path))
