#!/usr/bin/env python3
"""
CodeBLEU Comparison Script

Compares AI-generated patches against human patches using CodeBLEU metric.
Supports both Bigvul and Megavul datasets.

Usage:
    python compare_codebleu.py --patch-output patch_agent/outputs/patch_megavul_generated.jsonl \
                               --dataset processed_datasets/megavul_test.jsonl \
                               --output results/codebleu_comparison.json
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from codebleu import calc_codebleu
except ImportError:
    print("[WARNING] codebleu package not found. Install with: pip install codebleu")
    print("   Falling back to basic BLEU calculation only.")
    calc_codebleu = None

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except:
            try:
                nltk.download('punkt', quiet=True)
            except:
                pass
except ImportError:
    print("[WARNING] nltk not found. Install with: pip install nltk")
    sentence_bleu = None


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file into list of dictionaries."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def create_match_key(entry: Dict) -> str:
    """
    Create a matching key from entry fields.
    Tries multiple strategies for robust matching.
    """
    # Strategy 1: func_before hash (most reliable - should be identical)
    func_before = entry.get("func_before", "").strip()
    if func_before:
        # Use a hash of the normalized func_before for matching
        normalized = func_before.replace(" ", "").replace("\t", "").replace("\n", "")
        return f"func_hash::{hash(normalized[:500])}"
    
    # Strategy 2: func_name + file_path + repo_name (most specific)
    func_name = entry.get("func_name", "").strip()
    file_path = entry.get("file_path", "").strip()
    repo_name = entry.get("repo_name", "").strip()
    
    if func_name and file_path and repo_name:
        return f"{repo_name}::{file_path}::{func_name}"
    
    # Strategy 3: func_name + repo_name
    if func_name and repo_name:
        return f"{repo_name}::{func_name}"
    
    # Strategy 4: file_path + repo_name
    if file_path and repo_name:
        return f"{repo_name}::{file_path}"
    
    # Strategy 5: commit message hash
    commit_msg = entry.get("commit_msg") or entry.get("commit_message", "")
    if commit_msg:
        return f"commit::{hash(commit_msg[:100])}"
    
    return None


def match_entries(
    patch_entries: List[Dict],
    dataset_entries: List[Dict]
) -> List[Tuple[Dict, Dict]]:
    """
    Match patch entries with dataset entries to find human patches.
    Returns list of (patch_entry, dataset_entry) tuples.
    """
    # Build index of dataset entries by match key
    dataset_index = defaultdict(list)
    for entry in dataset_entries:
        key = create_match_key(entry)
        if key:
            dataset_index[key].append(entry)
    
    matched_pairs = []
    unmatched_patches = []
    
    for patch_entry in patch_entries:
        key = create_match_key(patch_entry)
        if not key:
            unmatched_patches.append(patch_entry)
            continue
        
        # Find best match
        candidates = dataset_index.get(key, [])
        if candidates:
            # If multiple candidates, prefer exact func_before match
            best_match = None
            patch_func_before = patch_entry.get("func_before", "").strip()
            
            for candidate in candidates:
                candidate_func_before = candidate.get("func_before", "").strip()
                # Normalize whitespace for comparison
                if patch_func_before.replace(" ", "").replace("\t", "") == candidate_func_before.replace(" ", "").replace("\t", ""):
                    best_match = candidate
                    break
            
            # If no exact match, use first candidate
            if not best_match:
                best_match = candidates[0]
            
            matched_pairs.append((patch_entry, best_match))
        else:
            unmatched_patches.append(patch_entry)
            # Debug: show why it didn't match
            if len(unmatched_patches) <= 3:
                print(f"   [DEBUG] Unmatched patch: func_name={patch_entry.get('func_name', 'N/A')}, repo={patch_entry.get('repo_name', 'N/A')}")
    
    print(f"[OK] Matched {len(matched_pairs)} pairs")
    if unmatched_patches:
        print(f"[WARNING] {len(unmatched_patches)} patch entries could not be matched")
    
    return matched_pairs


def compute_bleu(reference: str, candidate: str) -> float:
    """Compute BLEU score between reference and candidate code."""
    if sentence_bleu is None:
        return 0.0
    
    try:
        # Tokenize code (simple whitespace + punctuation split)
        ref_tokens = word_tokenize(reference.lower())
        cand_tokens = word_tokenize(candidate.lower())
        
        # Use smoothing to handle cases with no n-gram matches
        smoothing = SmoothingFunction().method1
        score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)
        return score
    except Exception as e:
        print(f"[WARNING] BLEU computation error: {e}")
        return 0.0


def compute_codebleu_score(reference: str, candidate: str, lang: str = "c") -> Dict[str, float]:
    """
    Compute CodeBLEU score between reference and candidate code.
    Returns dictionary with component scores.
    """
    if calc_codebleu is None:
        return {
            "codebleu": 0.0,
            "ngram_match_score": 0.0,
            "weighted_ngram_match_score": 0.0,
            "syntax_match_score": 0.0,
            "dataflow_match_score": 0.0,
            "error": "codebleu package not installed"
        }
    
    try:
        # Ensure code is not empty
        if not reference or not candidate:
            return {
                "codebleu": 0.0,
                "ngram_match_score": 0.0,
                "weighted_ngram_match_score": 0.0,
                "syntax_match_score": 0.0,
                "dataflow_match_score": 0.0,
                "error": "Empty reference or candidate code"
            }
        
        # Normalize language code
        lang_map = {"cpp": "cpp", "c++": "cpp", "c": "c"}
        lang = lang_map.get(lang.lower(), "c")
        
        # CodeBLEU expects lists of code strings
        references = [[reference]]  # List of reference lists
        predictions = [candidate]   # List of predictions
        
        # Call CodeBLEU
        result = calc_codebleu(
            references=references,
            predictions=predictions,
            lang=lang,
        )
        
        return {
            "codebleu": result.get("codebleu", 0.0),
            "ngram_match_score": result.get("ngram_match_score", 0.0),
            "weighted_ngram_match_score": result.get("weighted_ngram_match_score", 0.0),
            "syntax_match_score": result.get("syntax_match_score", 0.0),
            "dataflow_match_score": result.get("dataflow_match_score", 0.0),
        }
    except Exception as e:
        print(f"[WARNING] CodeBLEU computation error: {e}")
        return {
            "codebleu": 0.0,
            "ngram_match_score": 0.0,
            "weighted_ngram_match_score": 0.0,
            "syntax_match_score": 0.0,
            "dataflow_match_score": 0.0,
            "error": str(e)
        }


def compute_comparison_metrics(
    patch_entry: Dict,
    dataset_entry: Dict
) -> Dict:
    """
    Compute comparison metrics between AI patch and human patch.
    """
    # Try multiple field names for human patch
    human_patch = (
        dataset_entry.get("func_after", "") or 
        dataset_entry.get("func", "") or
        dataset_entry.get("patched_code", "")
    ).strip()
    
    ai_patch = patch_entry.get("patch_generated", "").strip()
    func_before = patch_entry.get("func_before", "").strip()
    
    # Diagnostic info
    missing_fields = []
    if not human_patch:
        missing_fields.append("human_patch (func_after)")
    if not ai_patch:
        missing_fields.append("ai_patch (patch_generated)")
    
    if not human_patch or not ai_patch:
        error_msg = f"Missing: {', '.join(missing_fields)}"
        if not human_patch:
            # Show available keys for debugging
            available_keys = [k for k in dataset_entry.keys() if 'func' in k.lower() or 'patch' in k.lower()]
            error_msg += f" | Available keys with 'func' or 'patch': {available_keys}"
        return {
            "error": error_msg,
            "codebleu": 0.0,
            "bleu": 0.0,
        }
    
    # Determine language from dataset entry or default to C
    lang = dataset_entry.get("lang", "c").lower()
    if lang not in ["c", "cpp", "c++"]:
        lang = "c"  # Default to C for CodeBLEU
    
    # Compute CodeBLEU
    codebleu_scores = compute_codebleu_score(human_patch, ai_patch, lang=lang)
    
    # Compute regular BLEU
    bleu_score = compute_bleu(human_patch, ai_patch)
    
    return {
        "func_name": patch_entry.get("func_name", ""),
        "file_path": patch_entry.get("file_path", ""),
        "repo_name": patch_entry.get("repo_name", ""),
        "commit_msg": patch_entry.get("commit_msg", "")[:100],  # Truncate for readability
        "codebleu": codebleu_scores.get("codebleu", 0.0),
        "codebleu_ngram": codebleu_scores.get("ngram_match_score", 0.0),
        "codebleu_weighted_ngram": codebleu_scores.get("weighted_ngram_match_score", 0.0),
        "codebleu_syntax": codebleu_scores.get("syntax_match_score", 0.0),
        "codebleu_dataflow": codebleu_scores.get("dataflow_match_score", 0.0),
        "bleu": bleu_score,
        "human_patch_length": len(human_patch),
        "ai_patch_length": len(ai_patch),
        "func_before_length": len(func_before),
        **codebleu_scores,  # Include any error messages
    }


def compute_aggregate_stats(results: List[Dict]) -> Dict:
    """Compute aggregate statistics from comparison results."""
    if not results:
        return {"error": "No results to aggregate"}
    
    # Filter out results with critical errors (missing data), but keep those with CodeBLEU errors if BLEU was computed
    valid_results = [
        r for r in results 
        if r.get("bleu", 0.0) > 0 or (r.get("codebleu", 0.0) > 0) or 
        (r.get("error", "") and "Missing patch data" not in r.get("error", ""))
    ]
    # If still no valid results, include any with BLEU scores
    if not valid_results:
        valid_results = [r for r in results if r.get("bleu", 0.0) > 0]
    
    if not valid_results:
        return {"error": "No valid results to aggregate"}
    
    codebleu_scores = [r.get("codebleu", 0.0) for r in valid_results]
    bleu_scores = [r.get("bleu", 0.0) for r in valid_results]
    
    stats = {
        "total_samples": len(results),
        "valid_samples": len(valid_results),
        "codebleu": {
            "mean": statistics.mean(codebleu_scores) if codebleu_scores else 0.0,
            "median": statistics.median(codebleu_scores) if codebleu_scores else 0.0,
            "std": statistics.stdev(codebleu_scores) if len(codebleu_scores) > 1 else 0.0,
            "min": min(codebleu_scores) if codebleu_scores else 0.0,
            "max": max(codebleu_scores) if codebleu_scores else 0.0,
        },
        "bleu": {
            "mean": statistics.mean(bleu_scores) if bleu_scores else 0.0,
            "median": statistics.median(bleu_scores) if bleu_scores else 0.0,
            "std": statistics.stdev(bleu_scores) if len(bleu_scores) > 1 else 0.0,
            "min": min(bleu_scores) if bleu_scores else 0.0,
            "max": max(bleu_scores) if bleu_scores else 0.0,
        },
        "distribution": {
            "codebleu_high": sum(1 for s in codebleu_scores if s >= 0.8),
            "codebleu_medium": sum(1 for s in codebleu_scores if 0.5 <= s < 0.8),
            "codebleu_low": sum(1 for s in codebleu_scores if s < 0.5),
        }
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Compare AI-generated patches with human patches using CodeBLEU"
    )
    parser.add_argument(
        "--patch-output",
        type=str,
        default="patch_agent/outputs/patch_megavul_generated.jsonl",
        help="Path to patch agent output JSONL file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="processed_datasets/megavul_test.jsonl",
        help="Path to dataset JSONL file (Bigvul or Megavul) - must have func_after field",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/codebleu_comparison.json",
        help="Path to output JSON file with results",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        help="Optional: Path to output CSV file with per-sample results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional: Limit number of samples to process (for testing)",
    )
    
    args = parser.parse_args()
    
    print("[*] Loading patch outputs...")
    patch_entries = load_jsonl(args.patch_output)
    if args.limit:
        patch_entries = patch_entries[:args.limit]
    print(f"   Loaded {len(patch_entries)} patch entries")
    
    print("[*] Loading dataset...")
    dataset_entries = load_jsonl(args.dataset)
    print(f"   Loaded {len(dataset_entries)} dataset entries")
    
    # Check if dataset has func_after field
    if dataset_entries:
        sample_entry = dataset_entries[0]
        has_func_after = "func_after" in sample_entry
        print(f"   Dataset has 'func_after' field: {has_func_after}")
        if not has_func_after:
            print("   [WARNING] Dataset missing 'func_after' field!")
            print("   Available keys:", list(sample_entry.keys())[:10])
            print("   Use processed_datasets/megavul_test.jsonl or processed_datasets/bigvul_val.jsonl instead")
    
    print("[*] Matching entries...")
    matched_pairs = match_entries(patch_entries, dataset_entries)
    
    if not matched_pairs:
        print("[ERROR] No matched pairs found. Exiting.")
        return
    
    print("[*] Computing CodeBLEU scores...")
    results = []
    error_count = 0
    for idx, (patch_entry, dataset_entry) in enumerate(matched_pairs):
        if (idx + 1) % 10 == 0:
            print(f"   Processed {idx + 1}/{len(matched_pairs)} samples...")
        
        metrics = compute_comparison_metrics(patch_entry, dataset_entry)
        results.append(metrics)
        
        # Track errors
        if "error" in metrics and metrics.get("error"):
            error_count += 1
            if error_count <= 3:  # Show first 3 errors as examples
                print(f"   [WARNING] Sample {idx + 1} error: {metrics.get('error', 'Unknown')}")
    
    if error_count > 0:
        print(f"   [WARNING] Total errors: {error_count}/{len(matched_pairs)} samples")
    
    print("[*] Computing aggregate statistics...")
    stats = compute_aggregate_stats(results)
    
    # Prepare output
    output_data = {
        "config": {
            "patch_output": args.patch_output,
            "dataset": args.dataset,
            "total_patches": len(patch_entries),
            "total_dataset_entries": len(dataset_entries),
            "matched_pairs": len(matched_pairs),
        },
        "statistics": stats,
        "results": results,
    }
    
    # Save JSON output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"[OK] Results saved to {output_path}")
    
    # Save CSV if requested
    if args.output_csv:
        import csv
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        if results:
            fieldnames = list(results[0].keys())
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            print(f"[OK] CSV results saved to {csv_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total samples: {stats.get('total_samples', 0)}")
    print(f"Valid samples: {stats.get('valid_samples', 0)}")
    print(f"\nCodeBLEU Scores:")
    codebleu_stats = stats.get("codebleu", {})
    print(f"  Mean:   {codebleu_stats.get('mean', 0.0):.4f}")
    print(f"  Median: {codebleu_stats.get('median', 0.0):.4f}")
    print(f"  Std:    {codebleu_stats.get('std', 0.0):.4f}")
    print(f"  Range:  [{codebleu_stats.get('min', 0.0):.4f}, {codebleu_stats.get('max', 0.0):.4f}]")
    print(f"\nBLEU Scores:")
    bleu_stats = stats.get("bleu", {})
    print(f"  Mean:   {bleu_stats.get('mean', 0.0):.4f}")
    print(f"  Median: {bleu_stats.get('median', 0.0):.4f}")
    print(f"\nDistribution:")
    dist = stats.get("distribution", {})
    print(f"  High similarity (>=0.8):   {dist.get('codebleu_high', 0)}")
    print(f"  Medium similarity (0.5-0.8): {dist.get('codebleu_medium', 0)}")
    print(f"  Low similarity (<0.5):    {dist.get('codebleu_low', 0)}")
    print("="*60)


if __name__ == "__main__":
    main()

