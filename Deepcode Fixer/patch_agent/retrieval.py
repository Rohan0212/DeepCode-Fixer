import logging
from pathlib import Path
from typing import Dict, List, Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import PatchAgentConfig
from .datasets import load_jsonl
from .utils import truncate_text

logger = logging.getLogger(__name__)


def _format_entry_for_embedding(entry: Dict) -> str:
    commit_msg = entry.get("commit_msg") or entry.get("commit_message") or ""
    func_before = entry.get("func_before", "")
    func_after = entry.get("func_after") or entry.get("func", "")
    rca = entry.get("rca_explanation") or entry.get("rca_generated") or ""
    return f"{commit_msg}\n{func_before}\n{func_after}\n{rca}"


class PatchRetriever:
    """Retrieval helper that wraps SentenceTransformer + FAISS."""

    def __init__(self, config: PatchAgentConfig):
        self.config = config
        logger.info("Loading Big-Vul dataset from %s", config.bigvul_path)
        self.entries: List[Dict] = load_jsonl(config.bigvul_path)
        logger.info("Loaded %d reference entries", len(self.entries))
        self.embedder = SentenceTransformer(config.embed_model)
        self.embeddings = self._load_or_build_embeddings()
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        logger.info("FAISS index ready (dim=%d)", self.embeddings.shape[1])

    def _load_or_build_embeddings(self) -> np.ndarray:
        cache_path = Path(self.config.embeddings_cache_path)
        if cache_path.exists() and not self.config.rebuild_embeddings:
            logger.info("Loading cached embeddings from %s", cache_path)
            cached = np.load(cache_path)
            if cached.shape[0] == len(self.entries):
                return cached
            logger.warning(
                "Cached embeddings count (%d) != entry count (%d); rebuilding.",
                cached.shape[0],
                len(self.entries),
            )

        logger.info("Building embeddings for %d entries...", len(self.entries))
        texts = [_format_entry_for_embedding(e) for e in self.entries]
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        embeddings = np.asarray(embeddings, dtype="float32")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        logger.info("Embeddings cached to %s", cache_path)
        return embeddings

    def retrieve(self, entry: Dict, top_k: int) -> List[Dict]:
        if not self.entries:
            return []
        query_text = _format_entry_for_embedding(entry)
        q_vec = self.embedder.encode([query_text], show_progress_bar=False)
        q_vec = np.asarray(q_vec, dtype="float32")
        k = min(top_k, len(self.entries))
        distances, indices = self.index.search(q_vec, k)
        results = [self.entries[idx] for idx in indices[0]]
        for res, score in zip(results, distances[0]):
            res["_retrieval_score"] = float(score)
        return results

    @staticmethod
    def to_metadata(examples: Sequence[Dict], char_limit: int) -> List[Dict]:
        metadata = []
        for ex in examples:
            metadata.append(
                {
                    "commit_msg": ex.get("commit_msg") or ex.get("commit_message"),
                    "repo_name": ex.get("repo_name"),
                    "func_name": ex.get("func_name"),
                    "score": ex.get("_retrieval_score"),
                    "vulnerable_excerpt": truncate_text(
                        ex.get("func_before", ""), char_limit
                    ),
                    "patch_excerpt": truncate_text(
                        ex.get("func_after") or ex.get("func", ""), char_limit
                    ),
                }
            )
        return metadata


