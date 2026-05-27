import os
import json
from tqdm import tqdm

# Input files
TRAIN_PATH = "processed_datasets/bigvul_train.jsonl"
VAL_PATH = "processed_datasets/bigvul_val.jsonl"

# Output files
OUT_DIR = "processed_datasets/rca_prompts"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_TRAIN = os.path.join(OUT_DIR, "rca_bigvul_train.jsonl")
OUT_VAL = os.path.join(OUT_DIR, "rca_bigvul_val.jsonl")


def load_jsonl(path):
    """Read JSONL file line-by-line into a list of dicts"""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_instruction(record):
    """
    Convert one BigVul record into an instruction/response pair.
    The RCA agent is trained to analyze buggy code and explain the root cause.
    """
    buggy_func = record.get("func_before", "")
    summary = record.get("summary", "")
    commit_msg = record.get("commit_message", "")

    # Handle NaN / non-string fields safely
    if not isinstance(summary, str):
        summary = ""
    if not isinstance(commit_msg, str):
        commit_msg = ""
    if not isinstance(buggy_func, str) or not buggy_func.strip():
        return None

    # Construct the instruction (UPDATED to structured RCA format)
    prompt = (
    "You are an expert software vulnerability analyst.\n"
    "Given the following buggy C/C++ function, identify:\n"
    "1) The ROOT CAUSE of the vulnerability (what is wrong and why)\n"
    "2) The SINK (dangerous function / API / resource interaction)\n"
    "3) The DATA FLOW / PATH that leads to the vulnerability (if identifiable)\n"
    "Return output STRICTLY in this JSON format:\n\n"
    "{\n"
    "  \"root_cause\": \"<short sentence explaining why the issue happens>\",\n"
    "  \"sink\": \"<function or location where failure/overflow/bug manifests>\",\n"
    "  \"path\": \"<variables / code path enabling the bug>\",\n"
    "  \"summary\": \"<1–2 sentence explanation of the vulnerability>\"\n"
    "}\n\n"
    "Function:\n"
    f"{buggy_func}\n\n"
    "RCA:"
    )


    # Use summary first, else fallback to commit message
    # Force output to fit into the structured JSON format
    response = json.dumps({
        "root_cause": summary.strip() or commit_msg.strip() or "Unknown root cause.",
        "sink": "",
        "path": "",
        "summary": summary.strip() or commit_msg.strip() or "Bug fix details unavailable."
    })

    return {"prompt": prompt, "response": response}


def convert_dataset(in_path, out_path):
    """Convert full dataset into instruction-response pairs"""
    data = load_jsonl(in_path)
    print(f"Loaded {len(data)} records from {in_path}")

    pairs = []
    for rec in tqdm(data, desc=f"Building RCA prompts from {os.path.basename(in_path)}"):
        pair = build_instruction(rec)
        if pair:
            pairs.append(pair)

    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(f"✅ Saved {len(pairs)} RCA training pairs → {out_path}")
    return pairs


def main():
    train_pairs = convert_dataset(TRAIN_PATH, OUT_TRAIN)
    val_pairs = convert_dataset(VAL_PATH, OUT_VAL)

    print("\n📊 Summary:")
    print(f"Train samples: {len(train_pairs)}")
    print(f"Val samples:   {len(val_pairs)}")

    print("\n💡 Example prompt/response pair:")
    example = train_pairs[0]
    print("\n--- Prompt ---\n", example["prompt"][:500], "...")
    print("\n--- Response ---\n", example["response"])


if __name__ == "__main__":
    main()
