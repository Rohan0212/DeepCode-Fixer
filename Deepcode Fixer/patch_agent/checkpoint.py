"""Checkpointing utilities for resuming long-running jobs."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint state for resuming interrupted jobs."""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.processed_ids: Set[str] = set()
        self._load()

    def _load(self) -> None:
        """Load existing checkpoint if available."""
        if self.checkpoint_path.exists():
            try:
                with self.checkpoint_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.processed_ids = set(data.get("processed_ids", []))
                logger.info(
                    "Loaded checkpoint: %d entries already processed",
                    len(self.processed_ids),
                )
            except Exception as exc:
                logger.warning("Failed to load checkpoint: %s", exc)
                self.processed_ids = set()

    def is_processed(self, entry_id: str) -> bool:
        """Check if an entry has already been processed."""
        return entry_id in self.processed_ids

    def mark_processed(self, entry_id: str) -> None:
        """Mark an entry as processed."""
        self.processed_ids.add(entry_id)

    def save(self) -> None:
        """Save checkpoint to disk."""
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"processed_ids": list(self.processed_ids)}
        with self.checkpoint_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.debug("Checkpoint saved: %d entries", len(self.processed_ids))

    def clear(self) -> None:
        """Clear checkpoint (start fresh)."""
        self.processed_ids.clear()
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        logger.info("Checkpoint cleared")

    def get_entry_id(self, entry: Dict) -> str:
        """Generate a unique ID for an entry."""
        # Try multiple fields to create a stable ID
        candidates = [
            entry.get("cve_id"),
            entry.get("commit_hash"),
            entry.get("id"),
            f"{entry.get('repo_name', '')}:{entry.get('file_path', '')}:{entry.get('func_name', '')}",
        ]
        for candidate in candidates:
            if candidate:
                return str(candidate)
        # Fallback: hash the function code
        func_before = entry.get("func_before", "")
        return f"hash_{hash(func_before) & 0xFFFFFFFFFFFFFFFF}"

