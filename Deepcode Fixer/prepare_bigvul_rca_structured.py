import os, json
from tqdm import tqdm

TRAIN_PATH = "processed_datasets/bigvul_train.jsonl"
VAL_PATH   = "processed_datasets/bigvul_val.jsonl"

OUT_DIR = "processed_datasets/rca_prompts_structured"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_TRAIN = os.path.join(OUT_DIR, "rca_bigvul_structured_train.jsonl")
OUT_VAL   = os.path.join(OUT_DIR, "rca_bigvul_structured_val.jsonl")


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_instruction(record):
    """
    Build structured RCA JSON training example.
    """
    buggy = record.get("func_before", "")
    commit_msg = record.get("commit_message", "") or record.get("summary", "")

    if not buggy.strip():
        return None

    prompt = f"""
You are an RCA extraction agent.

Given the vulnerable code (with line numbers), return ONLY valid JSON:

{{
  "source": "<where input or untrusted data comes from>",
  "path": "<how the data flows to the vulnerable point>",
  "sink": "<where the bug becomes exploitable>",
  "summary": "<1 sentence describing the vulnerability>",
  "evidence_lines": [<line numbers>],
  "cwe_guess": "CWE-xxx",
  "confidence": 0-100
}}

RULES:
- DO NOT copy commit messages.
- DO NOT output explanation outside of JSON.
- If unsure: guess, do NOT leave fields empty.

CODE:
{buggy}
""".strip()

    # we don't use commit message as output anymore
    response = {
        "source": "",
        "path": "",
        "sink": "",
        "summary": "",
        "evidence_lines": [],
        "cwe_guess": "",
        "confidence": 0,
    }

    return {"prompt": prompt, "response": response}


def convert(in_path, out_path):
    data = load_jsonl(in_path)
    pairs = []

    for rec in tqdm(data, desc=f"Building RCA structured from {in_path}"):
        pair = build_instruction(rec)
        if pair:
            pairs.append(pair)

    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(f"✅ Saved {len(pairs)} structured RCA pairs → {out_path}")


if __name__ == "__main__":
    convert(TRAIN_PATH, OUT_TRAIN)
    convert(VAL_PATH, OUT_VAL)
