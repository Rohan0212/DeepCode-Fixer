import json
import logging
from pathlib import Path

from tqdm import tqdm

from .checkpoint import CheckpointManager
from .config import PatchAgentConfig
from .datasets import iter_jsonl
from .generator import PatchLLM
from .prompt_builder import build_patch_prompt
from .retrieval import PatchRetriever
from .validator import PatchValidator
from .validators import (
    InputDataValidator,
    RCAOutputValidator,
    RetrievalValidator,
    LLMResponseValidator,
    EnhancedPatchValidator,
)
from .quality import PatchQualityEvaluator

logger = logging.getLogger("patch_agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run() -> None:
    config = PatchAgentConfig()
    config.ensure_directories()
    retriever = PatchRetriever(config)
    llm = PatchLLM(config)
    validator = PatchValidator(config)
    quality = PatchQualityEvaluator()

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize checkpoint manager if checkpoint path is configured
    checkpoint = None
    if config.checkpoint_path:
        checkpoint = CheckpointManager(Path(config.checkpoint_path))
        logger.info("Checkpointing enabled: %s", config.checkpoint_path)

    stats = {
        "total": 0,
        "written": 0,
        "skipped": 0,
        "errors": 0,
        "already_processed": 0,
        "invalid_input": 0,
        "invalid_rca": 0,
        "invalid_retrieval": 0,
        "invalid_llm_response": 0,
        "invalid_patch": 0,
    }
    session_records = []
    enhanced_validator = EnhancedPatchValidator()

    # Open output file in append mode if resuming, write mode if starting fresh
    # Note: When resuming, we append to preserve previous work, but checkpoint prevents duplicates
    mode = "a" if (checkpoint and checkpoint.processed_ids) else "w"
    if mode == "a":
        logger.info("Resuming from checkpoint - appending to existing output file")
    with output_path.open(mode, encoding="utf-8") as sink:
        iterator = iter_jsonl(config.megavul_with_rca_path, limit=config.sample_limit)
        for idx, entry in enumerate(
            tqdm(iterator, desc="Patch Agent", total=config.sample_limit)
        ):
            stats["total"] += 1

            # Check if already processed (checkpointing)
            if checkpoint:
                entry_id = checkpoint.get_entry_id(entry)
                if checkpoint.is_processed(entry_id):
                    stats["already_processed"] += 1
                    logger.debug("Skipping already processed entry: %s", entry_id)
                    continue

            try:
                # Validate input data
                input_validation = InputDataValidator.validate(entry)
                if not input_validation.is_valid:
                    stats["invalid_input"] += 1
                    stats["skipped"] += 1
                    logger.warning(
                        "Skipping sample %s (invalid input): %s",
                        entry.get("func_name", "unknown"),
                        input_validation.reason
                    )
                    if input_validation.warnings:
                        logger.debug("Input warnings: %s", input_validation.warnings)
                    if checkpoint:
                        checkpoint.mark_processed(checkpoint.get_entry_id(entry))
                    continue
                if input_validation.warnings:
                    logger.debug("Input warnings for %s: %s", entry.get("func_name"), input_validation.warnings)
                
                # Validate RCA quality
                rca_text = entry.get("rca_generated", "")
                rca_validation = RCAOutputValidator.validate(rca_text)
                if not rca_validation.is_valid:
                    stats["invalid_rca"] += 1
                    logger.warning(
                        "Poor quality RCA for %s: %s (continuing anyway)",
                        entry.get("func_name", "unknown"),
                        rca_validation.reason
                    )
                if rca_validation.warnings:
                    logger.debug("RCA warnings for %s: %s", entry.get("func_name"), rca_validation.warnings)
                
                # Retrieve examples
                retrieved = retriever.retrieve(entry, config.retrieval_top_k)
                
                # Validate retrieval results
                retrieval_validation = RetrievalValidator.validate(retrieved, config.retrieval_top_k)
                if not retrieval_validation.is_valid:
                    stats["invalid_retrieval"] += 1
                    logger.warning(
                        "Invalid retrieval for %s: %s (continuing anyway)",
                        entry.get("func_name", "unknown"),
                        retrieval_validation.reason
                    )
                if retrieval_validation.warnings:
                    logger.debug("Retrieval warnings for %s: %s", entry.get("func_name"), retrieval_validation.warnings)
                
                # Build prompt and generate patch
                prompt, _ = build_patch_prompt(entry, retrieved, config)
                llm_output = llm.generate_patch(prompt)
                
                # Validate LLM response
                llm_validation = LLMResponseValidator.validate_patch_response(llm_output)
                if not llm_validation.is_valid:
                    stats["invalid_llm_response"] += 1
                    stats["skipped"] += 1
                    logger.warning(
                        "Skipping sample %s (invalid LLM response): %s",
                        entry.get("func_name", "unknown"),
                        llm_validation.reason
                    )
                    if llm_validation.warnings:
                        logger.debug("LLM response warnings: %s", llm_validation.warnings)
                    if checkpoint:
                        checkpoint.mark_processed(checkpoint.get_entry_id(entry))
                    continue
                if llm_validation.warnings:
                    logger.debug("LLM response warnings for %s: %s", entry.get("func_name"), llm_validation.warnings)
                
                # Normalize and validate patch
                normalized = validator.normalize_patch(llm_output["patch"])
                validation = validator.validate(normalized)
                if not validation.is_valid:
                    stats["invalid_patch"] += 1
                    stats["skipped"] += 1
                    logger.warning("Skipping sample %s: %s", entry.get("func_name"), validation.reason)
                    if checkpoint:
                        checkpoint.mark_processed(checkpoint.get_entry_id(entry))
                    continue
                
                # Enhanced patch validation
                enhanced_validation = enhanced_validator.validate_syntax(validation.normalized_patch)
                if not enhanced_validation.is_valid:
                    stats["invalid_patch"] += 1
                    stats["skipped"] += 1
                    logger.warning(
                        "Skipping sample %s (enhanced validation failed): %s",
                        entry.get("func_name", "unknown"),
                        enhanced_validation.reason
                    )
                    if checkpoint:
                        checkpoint.mark_processed(checkpoint.get_entry_id(entry))
                    continue
                if enhanced_validation.warnings:
                    logger.debug("Enhanced validation warnings for %s: %s", entry.get("func_name"), enhanced_validation.warnings)
                
                # Structure validation
                structure_validation = enhanced_validator.validate_structure(validation.normalized_patch)
                if structure_validation.warnings:
                    logger.debug("Structure warnings for %s: %s", entry.get("func_name"), structure_validation.warnings)

                diff = validator.diff(entry.get("func_before", ""), validation.normalized_patch)
                quality_metrics = quality.evaluate(
                    entry.get("func_before", ""),
                    validation.normalized_patch,
                    diff,
                )
                record = {
                    "func_name": entry.get("func_name"),
                    "file_path": entry.get("file_path"),
                    "repo_name": entry.get("repo_name"),
                    "commit_msg": entry.get("commit_msg") or entry.get("commit_message"),
                    "func_before": entry.get("func_before"),
                    "rca_generated": entry.get("rca_generated"),
                    "patch_generated": validation.normalized_patch,
                    "patch_rationale": llm_output.get("rationale", ""),
                    "patch_diff": diff,
                    "quality_metrics": quality_metrics.as_dict(),
                    "retrieved_examples": PatchRetriever.to_metadata(
                        retrieved, config.retrieval_example_char_limit
                    ),
                }
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                session_records.append(record)
                stats["written"] += 1

                # Mark as processed and save checkpoint periodically
                if checkpoint:
                    checkpoint.mark_processed(checkpoint.get_entry_id(entry))
                    if (idx + 1) % config.checkpoint_interval == 0:
                        checkpoint.save()
                        logger.debug("Checkpoint saved after %d entries", idx + 1)

            except Exception as exc:  # pylint: disable=broad-except
                stats["errors"] += 1
                logger.exception("Failed to process sample: %s", exc)
                # Don't mark as processed on error so it can be retried

    # Final checkpoint save
    if checkpoint:
        checkpoint.save()

    session_path = output_path.with_suffix(".session.jsonl")
    with session_path.open("w", encoding="utf-8") as session_file:
        for record in session_records:
            session_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Patch Agent finished: %s", stats)
    logger.info("Session output written to %s", session_path)
    logger.info("Validation breakdown: Invalid Input=%d, Invalid RCA=%d, Invalid Retrieval=%d, "
                "Invalid LLM Response=%d, Invalid Patch=%d",
                stats["invalid_input"], stats["invalid_rca"], stats["invalid_retrieval"],
                stats["invalid_llm_response"], stats["invalid_patch"])


if __name__ == "__main__":
    run()


