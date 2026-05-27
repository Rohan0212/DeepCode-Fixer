import json
from tqdm import tqdm

INPUT_FILE = "processed_datasets/rca_prompts/rca_megavul.jsonl"
OUTPUT_FILE = "processed_datasets/rca_prompts/rca_megavul_labeled.jsonl"

def extract_rca(commit_msg):
    """
    Use commit_msg directly as RCA explanation (trimmed). 
    Can be improved with GPT later.
    """
    return commit_msg.strip()

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:

        for line in tqdm(f_in, desc="✍️ Adding RCA explanation"):
            record = json.loads(line)
            record["rca_explanation"] = extract_rca(record["commit_msg"])
            f_out.write(json.dumps(record) + "\n")

    print(f"✅ Saved RCA-labeled dataset to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
