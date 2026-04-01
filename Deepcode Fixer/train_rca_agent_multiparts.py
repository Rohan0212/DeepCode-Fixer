
import os
import time
from dotenv import load_dotenv
import os

load_dotenv()  

api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
client = OpenAI(api_key=api_key)

from openai import OpenAI

# ---------------- CONFIG ----------------
BASE_MODEL = "gpt-4o-mini-2024-07-18"                  
DATA_DIR = "processed_datasets/rca_prompts_openai/parts"
VAL_PATH = "processed_datasets/rca_prompts_openai/openai_rca_val.jsonl"
SLEEP_TIME = 60  
# ----------------------------------------

client = OpenAI()

def upload_file(path: str):
    """Upload a file and return its file ID."""
    print(f"📤 Uploading {path} ...")
    with open(path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="fine-tune")
    print(f"✅ Uploaded {os.path.basename(path)} — ID: {file_obj.id}")
    return file_obj.id

def start_finetune(train_id, val_id, suffix):
    """Start fine-tuning job and return the job ID."""
    print(f"🚀 Starting fine-tune on {suffix} ...")
    job = client.fine_tuning.jobs.create(
        training_file=train_id,
        validation_file=val_id,
        model=BASE_MODEL,
        suffix=f"rca-{suffix}"
    )
    print(f"✅ Fine-tune job started — ID: {job.id}")
    return job.id

def main():
    print("=== RCA Agent Multi-Part Fine-Tune Launcher ===")

    # Upload validation file once
    val_file_id = upload_file(VAL_PATH)

    # List training parts
    parts = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".jsonl")])
    if not parts:
        print("❌ No dataset parts found. Please check the directory.")
        return

    print(f"📦 Found {len(parts)} training parts → {parts}")

    # Loop over parts and train sequentially
    for idx, part in enumerate(parts, start=1):
        part_path = os.path.join(DATA_DIR, part)
        train_file_id = upload_file(part_path)

        job_id = start_finetune(train_file_id, val_file_id, f"part{idx}")

        print(f"⏳ Waiting {SLEEP_TIME} seconds before next upload...\n")
        time.sleep(SLEEP_TIME)

    print("\n🎯 All fine-tune jobs have been launched successfully!")
    print("👉 You can monitor them with: openai api fine_tuning.jobs.list")
    print("👉 Each job will produce a fine-tuned model ID once completed.")

if __name__ == "__main__":
    main()
