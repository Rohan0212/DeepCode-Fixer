import json
import os
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# ---------------- CONFIG ----------------
INPUT_FILE = "processed_datasets/rca_prompts/rca_bigvul_cleaned.jsonl"
OUTPUT_FILE = "processed_datasets/rca_prompts/rca_bigvul_augmented.jsonl"

# Set to False for dry-run (no GPT cost)
USE_GPT = True
load_dotenv()  

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)
# Use the "gpt-4o-mini" model for cost efficiency
MODEL_NAME = "gpt-4o-mini"


# ----------------------------------------

RCA_PROMPT_TEMPLATE = """You are an expert software vulnerability analyst.

Given the following buggy (vulnerable) and fixed (patched) C/C++ functions, generate a concise *Root Cause Analysis (RCA)* explaining:
1. The **root cause** (what mistake caused the vulnerability)
2. The **impact** (what could go wrong)
3. The **fix** (what change mitigates it)

Return your answer as a short paragraph (3–4 sentences max), clear and factual.

Buggy Function:
{func_before}

Fixed Function:
{func_after}

Commit Message (for context):
{commit_msg}

RCA Explanation:"""

def generate_rca_with_gpt(func_before, func_after, commit_msg):
    """Query GPT to generate an RCA explanation."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user",
                       "content": RCA_PROMPT_TEMPLATE.format(
                           func_before=func_before[:1500],
                           func_after=func_after[:1500],
                           commit_msg=commit_msg[:500]
                       )}],
            temperature=0.3,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ GPT error: {e}")
        return ""

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    print(f"📥 Loading cleaned Big-Vul dataset from: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    print(f"🧠 Processing {len(data)} entries...")
    augmented = []

    for entry in tqdm(data, desc="Generating RCA"):
        func_before = entry.get("func_before", "")
        func_after = entry.get("func_after", "")
        commit_msg = entry.get("commit_message", "")

        if not func_before or not func_after:
            continue

        if USE_GPT:
            rca_text = generate_rca_with_gpt(func_before, func_after, commit_msg)
        else:
            # Dry-run placeholder
            rca_text = f"[DRY-RUN] RCA would be generated here for commit: {commit_msg[:80]}..."

        entry["rca_explanation"] = rca_text
        augmented.append(entry)

    print(f"✅ RCA generation complete for {len(augmented)} entries")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for e in augmented:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"💾 Saved augmented dataset to: {OUTPUT_FILE}")
    print("Mode:", "LIVE (GPT-enabled)" if USE_GPT else "DRY RUN (no cost)")

if __name__ == "__main__":
    main()
