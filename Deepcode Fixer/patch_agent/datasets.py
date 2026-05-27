import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional


def iter_jsonl(path: str, limit: Optional[int] = None) -> Iterator[Dict]:
    """Yield JSON objects from a JSONL file."""
    with open(path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_jsonl(path: str, limit: Optional[int] = None) -> List[Dict]:
    """Load a JSONL file into memory."""
    return list(iter_jsonl(path, limit=limit))


def count_lines(path: str) -> int:
    """Count lines in a file efficiently."""
    with open(path, "rb") as handle:
        return sum(1 for _ in handle)


def ensure_parent(path: str) -> None:
    """Ensure parent directory exists."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


