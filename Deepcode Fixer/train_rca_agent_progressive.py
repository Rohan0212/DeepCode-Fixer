import os
import time
from openai import OpenAI
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()   # <-- THIS loads OPENAI_API_KEY from .env

client = OpenAI()


TRAIN_DIR = "processed_datasets/rca_prompts_openai/parts"
VAL_PATH = "processed_datasets/rca_prompts_openai/openai_rca_val.jsonl"

MODEL_BASE = "gpt-4o-mini-2024-07-18"
client = OpenAI()


def upload_file(path, retries=3):
    """Upload file with retry handling for 504/timeout errors."""
    for attempt in range(1, retries + 1):
        try:
            print(f"📤 Uploading {path} (attempt {attempt}/{retries})...")
            with open(path, "rb") as f:
                resp = client.files.create(file=f, purpose="fine-tune")
            print(f"✅ Uploaded → {resp.id}")
            return resp.id

        except Exception as e:
            print(f"⚠️ Upload failed: {e}")

            if attempt < retries:
                print("⏳ Waiting 10 seconds before retrying...")
                time.sleep(10)
            else:
                print("❌ Giving up after max retries")
                raise


def start_finetune(train_file_id, val_file_id, base_model, suffix):
    print(f"\n🚀 Starting fine-tune: {suffix}")
    job = client.fine_tuning.jobs.create(
        training_file=train_file_id,
        validation_file=val_file_id,
        model=base_model,
        suffix=suffix,
    )
    print(f"🔧 Job ID: {job.id}")
    return job.id

def wait_until_done(job_id):
    print(f"⏳ Waiting for job {job_id} to finish...")
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        print(f"   → Status: {status}")

        if status in ["succeeded", "failed", "cancelled"]:
            print(f"\n✅ Final status: {status}")
            return job

        time.sleep(30)

def main():
    print("\n=== RCA Progressive Fine-tuning (Stage 2) ===")

    # Upload validation file
    val_file_id = upload_file(VAL_PATH)

    # Get all split parts
    train_parts = sorted([f for f in os.listdir(TRAIN_DIR) if f.endswith(".jsonl")])
    if not train_parts:
        print("❌ No training parts found!")
        return

    model_to_use = MODEL_BASE
    trained_models = []

    for idx, part in enumerate(train_parts, start=1):
        part_path = os.path.join(TRAIN_DIR, part)
        train_file_id = upload_file(part_path)

        job_id = start_finetune(
            train_file_id=train_file_id,
            val_file_id=val_file_id,
            base_model=model_to_use,
            suffix=f"rca-v2-p{idx}",
        )

        job_result = wait_until_done(job_id)

        if job_result.fine_tuned_model:
            model_to_use = job_result.fine_tuned_model
            trained_models.append(model_to_use)
            print(f"✅ Updated base model for next round: {model_to_use}")

    # Save final model
    with open("rca_progressive_models.log", "w") as f:
        for m in trained_models:
            f.write(m + "\n")

    print("\n🎉 Progressive training completed!")
    print(f"Final model: {model_to_use}")

if __name__ == "__main__":
    main()
