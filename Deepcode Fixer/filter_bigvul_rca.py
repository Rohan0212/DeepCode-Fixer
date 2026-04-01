import json
import os
from tqdm import tqdm

INPUT_FILE = "processed_datasets/rca_prompts/rca_bigvul_filtered.jsonl"
OUTPUT_FILE = "processed_datasets/rca_prompts/rca_bigvul_cleaned.jsonl"

def is_meaningful_code(func: str) -> bool:
    """Check that the code is long enough and not trivial."""
    if not func or not isinstance(func, str):
        return False
    func = func.strip()
    # Must be at least 5 lines and 100 characters
    return len(func.splitlines()) >= 5 and len(func) > 100

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ File not found: {INPUT_FILE}")
        return

    print(f"📥 Loading entries from: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    print(f"🔍 Filtering Big-Vul entries ({len(data)} total)...")
    cleaned = []
    for entry in tqdm(data, desc="Filtering"):
        func_before = entry.get("func_before", "")
        func_after = entry.get("func_after", "")
        msg = entry.get("commit_message", "")

        # Skip if code or message invalid
        if not is_meaningful_code(func_before) or not is_meaningful_code(func_after):
            continue
        if not msg or len(msg.strip()) < 15:
            continue

        # Skip if before/after code is almost identical
        if func_before.strip() == func_after.strip():
            continue

        # Skip trivial boilerplate commit messages
        low_value_keywords = ["update", "merge", "sync", "revert", "readme", "test", "typo"]
        if any(k in msg.lower() for k in low_value_keywords):
            continue

        cleaned.append(entry)

    print(f"✅ Retained {len(cleaned)} high-quality entries out of {len(data)}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for e in cleaned:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"💾 Saved cleaned dataset to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
