import json
import logging
import os
import random
import time

import faiss
import numpy as np
import tqdm
from dotenv import load_dotenv
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from sentence_transformers import SentenceTransformer

from .validators import RCAAgentValidator

load_dotenv()

logger = logging.getLogger("rca_agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --------------------------
# CONFIG (env-overridable)
# --------------------------
DEFAULTS = {
    "bigvul_path": "processed_datasets/rca_bigvul_augmented.jsonl",
    "megavul_path": "processed_datasets/rca_megavul_cleaned.jsonl",
    "output_path": "outputs/rca_megavul_generated.jsonl",
    "embed_model": "all-MiniLM-L6-v2",
    "llm_model": "gpt-4o-mini",
    "top_k": 5,
    "sample_limit": None,
    "max_retries": 3,
    "retry_initial_delay": 1.0,
    "retry_max_delay": 60.0,
}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _retry_api_call(func, max_retries=3, initial_delay=1.0, max_delay=60.0, operation_name="API call"):
    """Execute API call with exponential backoff retry logic."""
    delay = initial_delay
    last_exception = None
    retryable_exceptions = (APIError, APIConnectionError, APITimeoutError, Exception)

    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except retryable_exceptions as exc:
            last_exception = exc
            if attempt < max_retries:
                logger.warning(
                    "%s failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    operation_name,
                    attempt,
                    max_retries,
                    str(exc)[:200],
                    delay,
                )
                time.sleep(delay)
                delay = min(delay * 2.0, max_delay)
            else:
                logger.error(
                    "%s failed after %d attempts: %s",
                    operation_name,
                    max_retries,
                    str(exc)[:200],
                )

    raise last_exception


def _load_config():
    """Read configuration from the current environment each run."""
    return {
        "bigvul_path": os.getenv("RCA_AGENT_BIGVUL_PATH", DEFAULTS["bigvul_path"]),
        "megavul_path": os.getenv("RCA_AGENT_MEGAVUL_PATH", DEFAULTS["megavul_path"]),
        "output_path": os.getenv("RCA_AGENT_OUTPUT_PATH", DEFAULTS["output_path"]),
        "embed_model": os.getenv("RCA_AGENT_EMBED_MODEL", DEFAULTS["embed_model"]),
        "llm_model": os.getenv("RCA_AGENT_MODEL", DEFAULTS["llm_model"]),
        "top_k": int(os.getenv("RCA_AGENT_TOP_K", str(DEFAULTS["top_k"]))),
        "sample_limit": os.getenv("RCA_AGENT_SAMPLE_LIMIT"),
        "max_retries": int(os.getenv("RCA_AGENT_MAX_RETRIES", str(DEFAULTS["max_retries"]))),
        "retry_initial_delay": float(os.getenv("RCA_AGENT_RETRY_DELAY", str(DEFAULTS["retry_initial_delay"]))),
        "retry_max_delay": float(os.getenv("RCA_AGENT_RETRY_MAX_DELAY", str(DEFAULTS["retry_max_delay"]))),
    }


def build_prompt(retrieved, new_commit_msg, func_before, func_after):
    context = ""
    for i, ex in enumerate(retrieved, 1):
        context += (
            f"Example {i}:\n"
            f"Commit Message: {ex.get('commit_msg', ex.get('commit_message', ''))}\n"
            f"Vulnerable Function (before fix):\n{ex.get('func_before', '')[:500]}\n\n"
            f"Fixed Function (after patch):\n{ex.get('func_after', ex.get('func', ''))[:500]}\n\n"
            f"RCA Explanation: {ex.get('rca_explanation', '(missing)')}\n\n"
        )

    return f"""
You are an expert software security analyst. Given a commit message and vulnerable code diff, 
explain the *root cause* of the vulnerability in a concise paragraph.

Here are similar examples with their RCA explanations:
{context}
Now analyze this new commit and generate an RCA explanation:

Commit Message:
{new_commit_msg}

Vulnerable Function (before fix):
{func_before[:1000]}

Fixed Function (after patch):
{func_after[:1000]}
"""


def _determine_sample(entries, sample_limit):
    if not entries:
        return []
    limit = None if sample_limit is None else max(0, int(sample_limit))
    if limit is None:
        return entries
    if limit <= 0 or limit >= len(entries):
        return entries
    return random.sample(entries, limit)


def main():
    cfg = _load_config()
    logger.info("[*] Loading Big-Vul RCA dataset...")
    with open(cfg["bigvul_path"], encoding="utf-8") as handle:
        bigvul_entries = [json.loads(line) for line in handle]
    embedder = SentenceTransformer(cfg["embed_model"])

    texts = [
        (
            ex.get("commit_msg", ex.get("commit_message", "")) + "\n"
            + ex.get("func_before", "") + "\n"
            + ex.get("func_after", ex.get("func", ""))
        )
        for ex in bigvul_entries
    ]
    embeddings = embedder.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    logger.info("[OK] FAISS index built with %d examples", len(bigvul_entries))

    logger.info("[*] Loading Mega-Vul dataset...")
    with open(cfg["megavul_path"], encoding="utf-8") as handle:
        megavul_entries = [json.loads(line) for line in handle]

    entries_to_process = _determine_sample(megavul_entries, cfg["sample_limit"])
    logger.info("[*] Processing %d samples", len(entries_to_process))

    stats = {"total": 0, "valid_inputs": 0, "valid_outputs": 0, "skipped": 0, "errors": 0}
    
    with open(cfg["output_path"], "w", encoding="utf-8") as out_file:
        for ex in tqdm.tqdm(entries_to_process):
            stats["total"] += 1
            
            # Validate input
            input_validation = RCAAgentValidator.validate_input(ex)
            if not input_validation.is_valid:
                stats["skipped"] += 1
                logger.warning(
                    "Skipping entry (invalid input): %s - %s",
                    ex.get("id", "unknown"),
                    input_validation.reason
                )
                if input_validation.warnings:
                    logger.debug("Input warnings: %s", input_validation.warnings)
                continue
            
            stats["valid_inputs"] += 1
            if input_validation.warnings:
                logger.debug("Input warnings for entry %s: %s", ex.get("id", "unknown"), input_validation.warnings)
            
            query_text = (
                ex.get("commit_msg", ex.get("commit_message", ""))
                + "\n"
                + ex.get("func_before", "")
            )
            q_vec = embedder.encode([query_text]).astype("float32")
            _, indices = index.search(q_vec, min(cfg["top_k"], len(bigvul_entries)))

            retrieved = [bigvul_entries[i] for i in indices[0]]
            prompt = build_prompt(
                retrieved,
                ex.get("commit_msg", ex.get("commit_message", "")),
                ex.get("func_before", ""),
                ex.get("func_after", ex.get("func", "")),
            )

            try:
                def _call_api():
                    return client.chat.completions.create(
                        model=cfg["llm_model"],
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an expert RCA reasoning model.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.4,
                        max_tokens=400,
                    )

                resp = _retry_api_call(
                    _call_api,
                    max_retries=cfg.get("max_retries", 3),
                    initial_delay=cfg.get("retry_initial_delay", 1.0),
                    max_delay=cfg.get("retry_max_delay", 60.0),
                    operation_name=f"RCA generation ({cfg['llm_model']})",
                )
                rca_text = resp.choices[0].message.content.strip()
                
                # Validate output
                output_validation = RCAAgentValidator.validate_output(rca_text)
                if not output_validation.is_valid:
                    stats["skipped"] += 1
                    logger.warning(
                        "Skipping entry (invalid RCA output): %s - %s",
                        ex.get("id", "unknown"),
                        output_validation.reason
                    )
                    if output_validation.warnings:
                        logger.debug("Output warnings: %s", output_validation.warnings)
                    # Still save but mark as invalid
                    ex["rca_generated"] = rca_text
                    ex["rca_validation_status"] = "invalid"
                    ex["rca_validation_reason"] = output_validation.reason
                else:
                    stats["valid_outputs"] += 1
                    if output_validation.warnings:
                        logger.debug("Output warnings for entry %s: %s", ex.get("id", "unknown"), output_validation.warnings)
                    ex["rca_generated"] = rca_text
                    ex["rca_validation_status"] = "valid"
                
            except Exception as err:  # pylint: disable=broad-except
                stats["errors"] += 1
                logger.error("Error generating RCA for entry %s: %s", ex.get("id", "unknown"), str(err))
                rca_text = f"[ERROR] {err}"
                ex["rca_generated"] = rca_text
                ex["rca_validation_status"] = "error"

            out_file.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    logger.info("RCA Agent completed: %s", stats)

    logger.info("[OK] RCA explanations saved to %s", cfg["output_path"])
    logger.info("[*] Statistics: Total=%d, Valid Inputs=%d, Valid Outputs=%d, Skipped=%d, Errors=%d",
                stats["total"], stats["valid_inputs"], stats["valid_outputs"], stats["skipped"], stats["errors"])


if __name__ == "__main__":
    main()
