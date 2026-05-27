# Automated Security Patch Generation with Root Cause Analysis

## Abstract

This project implements a two-stage pipeline for automated security vulnerability patching:

1. **RCA Agent**: Generates root cause analyses for vulnerable code using retrieval-augmented generation
2. **Patch Agent**: Produces secure code patches based on RCA explanations using few-shot learning

The system processes datasets from Big-Vul and Mega-Vul, generates explanations and patches, and evaluates them using BLEU metrics against human-authored patches.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (for LLM access)
- 8GB+ RAM (for embedding models and FAISS indices)
- 10GB+ disk space (for datasets and embeddings)

- Download the datasets from this link and paste them in the directory: https://drive.google.com/file/d/1ImtT7q0qGHuIoCNBVTVwOGWW48rZGE-W/view?usp=sharing

### Setup

1. **Create a virtual environment:**

   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # Linux/Mac
   source .venv/bin/activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data (required for BLEU evaluation):**

   ```bash
   python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"
   ```

4. **Set up environment variables:**

   Create a `.env` file in the project root:

   ```env
   OPENAI_API_KEY=sk-your-api-key-here
   ```

   Or set it in your shell:

   ```bash
   # Windows PowerShell
   $env:OPENAI_API_KEY="sk-your-api-key-here"

   # Linux/Mac
   export OPENAI_API_KEY="sk-your-api-key-here"
   ```

5. **Verify installation:**
   ```bash
   python -c "import openai, faiss, sentence_transformers, nltk; print('All dependencies OK')"
   ```

---

## How to Run

### Quick Test Run (2 samples)

```bash
# Set environment variables
export OPENAI_API_KEY="sk-your-key"
export RCA_AGENT_SAMPLE_LIMIT=2
export PATCH_AGENT_SAMPLE_LIMIT=2

# Run full pipeline
python run_full_pipeline.py --skip-report
```

This will:

1. Generate RCA for 2 samples (~30 seconds)
2. Generate patches for 2 samples (~1 minute)
3. Skip PDF report generation

### Full Pipeline Run

```bash
# Set sample limits (optional, defaults to all)
export RCA_AGENT_SAMPLE_LIMIT=100
export PATCH_AGENT_SAMPLE_LIMIT=100

# Run complete pipeline
python run_full_pipeline.py
```

**Outputs:**

- `rca_agent/outputs/rca_megavul_generated.jsonl` - Generated RCA explanations
- `patch_agent/outputs/patch_megavul_generated.jsonl` - Generated patches
- `patch_agent/outputs/patch_report.pdf` - Comprehensive PDF report

### Run Individual Components

**RCA Agent only:**

```bash
python -m rca_agent.rca_agent
```

**Patch Agent only (requires RCA output):**

```bash
python -m patch_agent.run_patch_agent
```

**Evaluation:**

```bash
python compare_codebleu.py \
    --patch-output patch_agent/outputs/patch_megavul_generated.jsonl \
    --dataset processed_datasets/megavul_test.jsonl \
    --output results/codebleu_comparison.json
```

**Generate PDF Report:**

```bash
python -m patch_agent.report \
    --input patch_agent/outputs/patch_megavul_generated.jsonl
```

### Pipeline Options

```bash
# Only run RCA Agent
python run_full_pipeline.py --rca-only

# Only run Patch Agent (needs RCA output to exist)
python run_full_pipeline.py --patch-only

# Only generate report (needs patch output to exist)
python run_full_pipeline.py --report-only

# Full pipeline without report
python run_full_pipeline.py --skip-report
```

---

## Configuration

### Environment Variables

**Required:**

- `OPENAI_API_KEY` - Your OpenAI API key

### Missing Dependencies

**Solution:**

```bash
pip install -r requirements.txt
```

### NLTK Data Missing

**Solution:**

```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"
```
