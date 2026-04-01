import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _get_env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _get_env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val is not None else default


def _get_env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val is not None else default


def _get_env_optional_int(name: str) -> Optional[int]:
    val = os.getenv(name)
    return int(val) if val is not None else None


@dataclass
class PatchAgentConfig:
    """Centralised configuration for the Patch Agent."""

    bigvul_path: str = field(
        default_factory=lambda: os.getenv(
            "PATCH_AGENT_BIGVUL_PATH",
            "processed_datasets/rca_prompts/rca_bigvul_augmented.jsonl",
        )
    )
    megavul_with_rca_path: str = field(
        default_factory=lambda: os.getenv(
            "PATCH_AGENT_MEGAVUL_PATH",
            "rca_agent/outputs/rca_megavul_generated.jsonl",
        )
    )
    output_path: str = field(
        default_factory=lambda: os.getenv(
            "PATCH_AGENT_OUTPUT_PATH",
            "patch_agent/outputs/patch_megavul_generated.jsonl",
        )
    )

    embed_model: str = field(
        default_factory=lambda: os.getenv(
            "PATCH_AGENT_EMBED_MODEL", "all-MiniLM-L6-v2"
        )
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("PATCH_AGENT_MODEL", "gpt-4o-mini")
    )
    retrieval_top_k: int = field(
        default_factory=lambda: _get_env_int("PATCH_AGENT_TOP_K", 5)
    )
    temperature: float = field(
        default_factory=lambda: _get_env_float("PATCH_AGENT_TEMPERATURE", 0.15)
    )
    max_new_tokens: int = field(
        default_factory=lambda: _get_env_int("PATCH_AGENT_MAX_TOKENS", 900)
    )
    max_patch_chars: int = field(
        default_factory=lambda: _get_env_int("PATCH_AGENT_MAX_PATCH_CHARS", 6000)
    )
    retrieval_example_char_limit: int = field(
        default_factory=lambda: _get_env_int("PATCH_AGENT_EXAMPLE_CHAR_LIMIT", 800)
    )
    sample_limit: Optional[int] = field(
        default_factory=lambda: _get_env_optional_int("PATCH_AGENT_SAMPLE_LIMIT")
    )

    embeddings_cache_path: str = field(
        default_factory=lambda: os.getenv(
            "PATCH_AGENT_EMBED_CACHE", "patch_agent/artifacts/bigvul_embeddings.npy"
        )
    )
    rebuild_embeddings: bool = field(
        default_factory=lambda: _get_env_bool(
            "PATCH_AGENT_REBUILD_EMBEDDINGS", False
        )
    )

    system_prompt: str = field(
        default_factory=lambda: os.getenv(
            "PATCH_AGENT_SYSTEM_PROMPT",
            (
                "You are a senior security engineer. "
                "Produce minimal, production-ready fixes without altering unrelated logic."
            ),
        )
    )

    # Retry configuration
    max_retries: int = field(
        default_factory=lambda: _get_env_int("PATCH_AGENT_MAX_RETRIES", 3)
    )
    retry_initial_delay: float = field(
        default_factory=lambda: _get_env_float("PATCH_AGENT_RETRY_DELAY", 1.0)
    )
    retry_max_delay: float = field(
        default_factory=lambda: _get_env_float("PATCH_AGENT_RETRY_MAX_DELAY", 60.0)
    )

    # Checkpoint configuration
    checkpoint_path: Optional[str] = field(
        default_factory=lambda: os.getenv("PATCH_AGENT_CHECKPOINT_PATH")
    )
    checkpoint_interval: int = field(
        default_factory=lambda: _get_env_int("PATCH_AGENT_CHECKPOINT_INTERVAL", 10)
    )

    def ensure_directories(self) -> None:
        """Create directories required for outputs and caches."""
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.embeddings_cache_path).parent.mkdir(parents=True, exist_ok=True)
        if self.checkpoint_path:
            Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

