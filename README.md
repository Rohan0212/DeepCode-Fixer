# DeepCode-Fixer

Automated security patch generation using a **two-stage pipeline**:

- **RCA Agent**: retrieval-augmented Root Cause Analysis generation for vulnerable code
- **Patch Agent**: generates minimal, production-ready security patches guided by the RCA

This repo includes dataset preparation scripts, a full pipeline runner, and evaluation utilities.

## What’s in this repo

The main code lives in `Deepcode Fixer/`:

- `rca_agent/`: RCA generation (FAISS + SentenceTransformers + OpenAI)
- `patch_agent/`: patch generation + validation + report generation
- `run_full_pipeline.py`: orchestrates RCA → Patch → (optional) PDF report
- `compare_codebleu.py`: evaluates generated patches (CodeBLEU / related metrics)
- `processed_datasets/`: expected location for prepared datasets (see below)

## Requirements

- Python 3.8+
- An OpenAI API key (set via `OPENAI_API_KEY`)
- RAM/Disk: depends on dataset size (embeddings + FAISS index can be several GB)

## Setup

From the repo root:

```bash
cd "Deepcode Fixer"
python -m venv .venv
```

Activate the virtualenv:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download NLTK data (used by evaluation tooling):

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Datasets

This project expects prepared Big-Vul / Mega-Vul JSONL files under `processed_datasets/` (some scripts generate intermediate outputs under `processed_datasets/rca_prompts/`).

- Dataset bundle link: [Google Drive download](https://drive.google.com/file/d/1ImtT7q0qGHuIoCNBVTVwOGWW48rZGE-W/view?usp=sharing)

After downloading, place the extracted dataset folders under:

```text
Deepcode Fixer/processed_datasets/
```

## Configure your API key

Option A: set an environment variable:

```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."

# macOS / Linux
export OPENAI_API_KEY="sk-..."
```

Option B: create a `.env` file (recommended for local runs):

```env
OPENAI_API_KEY=sk-...
```

Important: **Do not commit** your `.env` file to GitHub.

## Quickstart

Run a tiny sample (useful for verifying everything works):

```bash
# Windows (PowerShell)
$env:RCA_AGENT_SAMPLE_LIMIT="2"
$env:PATCH_AGENT_SAMPLE_LIMIT="2"
python run_full_pipeline.py --skip-report
```

Run the full pipeline (RCA → Patch → PDF report):

```bash
python run_full_pipeline.py
```

### Run individual stages

RCA only:

```bash
python run_full_pipeline.py --rca-only
```

Patch only (assumes RCA output exists):

```bash
python run_full_pipeline.py --patch-only
```

Report only (assumes patch output exists):

```bash
python run_full_pipeline.py --report-only
```

## Outputs

Default output locations (can be overridden via env vars):

- `rca_agent/outputs/rca_megavul_generated.jsonl`
- `patch_agent/outputs/patch_megavul_generated.jsonl`
- `patch_agent/outputs/patch_report.pdf`

The Patch Agent also writes a session file next to the output:

- `patch_agent/outputs/patch_megavul_generated.session.jsonl`

## Configuration (common env vars)

RCA Agent:

- `RCA_AGENT_BIGVUL_PATH`: Big-Vul JSONL used for retrieval
- `RCA_AGENT_MEGAVUL_PATH`: Mega-Vul JSONL to generate RCA for
- `RCA_AGENT_OUTPUT_PATH`: where to write generated RCA JSONL
- `RCA_AGENT_MODEL`: LLM model name (default: `gpt-4o-mini`)
- `RCA_AGENT_TOP_K`: retrieval top-k (default: 5)
- `RCA_AGENT_SAMPLE_LIMIT`: limit number of processed samples

Patch Agent:

- `PATCH_AGENT_MEGAVUL_PATH`: input JSONL containing `rca_generated` (default: RCA output)
- `PATCH_AGENT_OUTPUT_PATH`: where to write patch outputs
- `PATCH_AGENT_MODEL`: LLM model name (default: `gpt-4o-mini`)
- `PATCH_AGENT_TOP_K`: retrieval top-k (default: 5)
- `PATCH_AGENT_SAMPLE_LIMIT`: limit number of processed samples
- `PATCH_AGENT_CHECKPOINT_PATH`: enable checkpointing (resume-friendly)

## Evaluation

Example (update paths if your filenames differ):

```bash
python compare_codebleu.py ^
  --patch-output patch_agent/outputs/patch_megavul_generated.jsonl ^
  --dataset processed_datasets/megavul_test.jsonl ^
  --output results/codebleu_comparison.json
```

## Troubleshooting

- **Auth errors**: ensure `OPENAI_API_KEY` is set in your shell (or in `.env`).
- **Missing datasets**: confirm `processed_datasets/` contains the JSONL files referenced by the env vars.
- **NLTK errors**: re-run the NLTK download command in the Setup section.
