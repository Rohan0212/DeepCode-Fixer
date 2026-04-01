import json
import os
from tqdm import tqdm

INPUT_FILE = "processed_datasets/megavul.jsonl"
OUTPUT_FILE = "processed_datasets/rca_prompts/rca_megavul_cleaned.jsonl"

BAD_PHRASES = [
    "update readme", "merge branch", "bump version", "typo", "style fix",
    "refactor", "reformat", "cleanup", "whitespace", "test:", "test="
]

def safe_str(value):
    """Convert None → '' and strip safely."""
    if isinstance(value, str):
        return value.strip()
    return ""

def is_valid(entry):
    func_before = safe_str(entry.get("func_before"))
    func_after = safe_str(entry.get("func"))
    commit_msg = safe_str(entry.get("commit_msg")).lower()
    cve_id = safe_str(entry.get("cve_id"))

    if not func_before or not func_after:
        return False
    if len(func_before) < 30 or len(func_after) < 30:
        return False
    if not commit_msg or len(commit_msg) < 25:
        return False
    if any(bad in commit_msg for bad in BAD_PHRASES):
        return False
    if not cve_id:
        return False

    return True

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = []
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip malformed lines

    print(f"📥 Loaded {len(data)} entries from Mega-Vul")

    valid_entries = [entry for entry in tqdm(data, desc="🔍 Filtering valid RCA entries") if is_valid(entry)]
    print(f"✅ Retained {len(valid_entries)} valid RCA entries")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for e in valid_entries:
            f.write(json.dumps(e) + "\n")

    print(f"💾 Saved cleaned RCA dataset to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
