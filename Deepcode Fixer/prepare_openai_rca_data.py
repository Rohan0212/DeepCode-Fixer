import json
from pathlib import Path
from tqdm import tqdm

IN_DIR = Path("processed_datasets/rca_prompts")
OUT_DIR = Path("processed_datasets/rca_prompts_openai")
OUT_DIR.mkdir(exist_ok=True, parents=True)

def convert_for_openai():
    files = [
        ("rca_bigvul_train.jsonl", "openai_rca_train.jsonl"),
        ("rca_bigvul_val.jsonl", "openai_rca_val.jsonl")
    ]

    for infile, outfile in files:
        in_path = IN_DIR / infile
        out_path = OUT_DIR / outfile
        print(f"🔄 Converting {in_path} → {out_path}")

        converted = []
        with open(in_path, "r", encoding="utf-8") as fin:
            for line in tqdm(fin, total=sum(1 for _ in open(in_path, "r", encoding="utf-8"))):
                record = json.loads(line)

                # Detect which fields are present
                prompt = record.get("instruction") or record.get("prompt", "")
                input_code = record.get("input") or ""
                output_text = record.get("output") or record.get("response", "")

                if not (input_code or output_text):
                    continue

                messages = [
                    {"role": "system", "content": "You are an expert code analyst specializing in software vulnerabilities."},
                    {"role": "user", "content": f"{prompt}\n\nCode:\n{input_code}"},
                    {"role": "assistant", "content": output_text.strip()}
                ]
                converted.append({"messages": messages})

        with open(out_path, "w", encoding="utf-8") as fout:
            for entry in converted:
                fout.write(json.dumps(entry) + "\n")

        print(f"✅ Saved {len(converted)} formatted samples → {out_path}")

if __name__ == "__main__":
    convert_for_openai()
