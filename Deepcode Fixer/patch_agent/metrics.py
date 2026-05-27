#!/usr/bin/env python3
"""
Utility to compute aggregate metrics over Patch Agent outputs.
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List


def load_records(path: Path) -> List[Dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def summarize(records: List[Dict]) -> Dict:
    total = len(records)
    if total == 0:
        return {"total": 0}

    scores = [
        rec.get("quality_metrics", {}).get("score")
        for rec in records
        if rec.get("quality_metrics")
    ]
    avg_score = statistics.mean(scores) if scores else None
    median_score = statistics.median(scores) if scores else None

    avg_added_lines = statistics.mean(
        rec.get("quality_metrics", {}).get("added_lines", 0) for rec in records
    )
    avg_deleted_lines = statistics.mean(
        rec.get("quality_metrics", {}).get("deleted_lines", 0) for rec in records
    )

    secure_replacements = sum(
        1
        for rec in records
        if rec.get("quality_metrics", {}).get("unsafe_removed")
        and rec.get("quality_metrics", {}).get("safe_added")
    )

    # Success rate: records with valid patches
    valid_patches = sum(
        1
        for rec in records
        if rec.get("patch_generated") and rec.get("patch_generated") != ""
    )
    success_rate = (valid_patches / total * 100) if total > 0 else 0.0

    # Quality distribution
    high_quality = sum(1 for s in scores if s and s >= 0.8)
    medium_quality = sum(1 for s in scores if s and 0.5 <= s < 0.8)
    low_quality = sum(1 for s in scores if s and s < 0.5)

    # Common patterns
    bounds_checks_added = sum(
        1
        for rec in records
        if rec.get("quality_metrics", {}).get("bounds_check_added")
    )
    unsafe_removed = sum(
        1 for rec in records if rec.get("quality_metrics", {}).get("unsafe_removed")
    )
    safe_added = sum(
        1 for rec in records if rec.get("quality_metrics", {}).get("safe_added")
    )

    # Failure modes (from RCA or patch generation errors)
    failure_modes = {}
    for rec in records:
        rca = rec.get("rca_generated", "")
        patch = rec.get("patch_generated", "")
        if "[ERROR]" in rca:
            failure_modes["RCA Generation Error"] = (
                failure_modes.get("RCA Generation Error", 0) + 1
            )
        if not patch or patch == "":
            failure_modes["Empty Patch"] = failure_modes.get("Empty Patch", 0) + 1
        elif "error" in patch.lower() or "failed" in patch.lower():
            failure_modes["Patch Generation Error"] = (
                failure_modes.get("Patch Generation Error", 0) + 1
            )

    stats = {
        "total_records": total,
        "valid_patches": valid_patches,
        "success_rate": round(success_rate, 2),
        "avg_quality_score": round(avg_score, 3) if avg_score is not None else None,
        "median_quality_score": round(median_score, 3)
        if median_score is not None
        else None,
        "quality_distribution": {
            "high_quality (≥0.8)": high_quality,
            "medium_quality (0.5-0.8)": medium_quality,
            "low_quality (<0.5)": low_quality,
        },
        "avg_added_lines": round(avg_added_lines, 2),
        "avg_deleted_lines": round(avg_deleted_lines, 2),
        "secure_replacements": secure_replacements,
        "security_improvements": {
            "bounds_checks_added": bounds_checks_added,
            "unsafe_calls_removed": unsafe_removed,
            "safe_alternatives_added": safe_added,
        },
        "failure_modes": failure_modes if failure_modes else None,
    }
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics for Patch Agent output JSONL."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("patch_agent/outputs/patch_megavul_generated.jsonl"),
        help="Path to the Patch Agent JSONL output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the metrics as JSON.",
    )
    args = parser.parse_args()

    records = load_records(args.input)
    stats = summarize(records)

    print(json.dumps(stats, indent=2, ensure_ascii=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()


