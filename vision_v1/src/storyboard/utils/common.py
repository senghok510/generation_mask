from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TypeVar

import numpy as np
import torch

T = TypeVar("T")


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(payload: dict, path: str | Path) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Sequence[dict], path: str | Path) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(prefer_half: bool = False) -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16 if prefer_half else torch.float32
    if torch.backends.mps.is_available():
        return "mps", torch.float16 if prefer_half else torch.float32
    return "cpu", torch.float32


def chunked(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def flatten(nested: Iterable[Iterable[T]]) -> list[T]:
    return [item for group in nested for item in group]
