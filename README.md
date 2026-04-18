
This repository contains the necessary code to reproduce the results of our submitted paper to EACL 2026 ***"Evidential Semantic Entropy for LLM Uncertainty Quantification"***.

This code is build on [kernel language entropy codebase](https://github.com/AlexanderVNikitin/kernel-language-entropy), which itself is build on [Semantic Uncertainty codebase](https://github.com/jlko/semantic_uncertainty/tree/master). 

---

## Project Structure

```
EvidentialSemanticEntropy/
├── README.md
├── setup.py                          # Package setup (package name: kle)
├── environment.yml                   # Conda environment
├── configs/
│   └── data_config.yaml             # Device & HuggingFace cache config
├── evsme/                           # Evidential Semantic Entropy module (new)
│   ├── evidence_theory_utils.py
│   ├── evidential_entropies.py
│   ├── evidential_framework_construction.py
│   └── other_uq.py
├── kle/                             # Kernel Language Entropy module
│   ├── core.py
│   ├── kernels.py
│   └── utils.py
└── semantic_uncertainty/            # Main pipeline scripts
    ├── generate_answers.py          # Entry point: generation + uncertainty
    ├── compute_uncertainty_measures.py
    ├── analyze_results.py
    └── uncertainty/
        ├── data/data_utils.py
        ├── models/huggingface_models.py
        ├── uncertainty_measures/
        └── utils/utils.py           # Argument parser (all CLI flags)
```

---

## Installation

**Step 1 - Create the Conda environment:**

```bash
conda env create -f environment.yml
```

**Step 2 - Activate the environment:**

```bash
conda activate evsme
```

> **Note:** The codebase requires CUDA-capable GPUs. `environment.yml` pins CUDA 12 libraries (`nvidia-*-cu12`) and installs this package in editable mode. Multi-GPU setups are supported via `configs/data_config.yaml` (see below).

---

## Prerequisites

| Requirement | Details |
|---|---|
| Python | 3.x |
| PyTorch | 2.7.0 (CUDA 12) |
| GPU | Recommended; multi-GPU supported |
| wandb | Account required for experiment logging |
| HuggingFace token | Required for gated models (Llama-2, etc.) |

**Set your wandb entity** (optional but recommended):

```bash
export WANDB_SEM_UNC_ENTITY=your_wandb_username
```

**Set your HuggingFace token** for gated models:

```bash
huggingface-cli login
```

---

## Running Experiments

### Main Entry Point

All experiments run through `semantic_uncertainty/generate_answers.py` from the **repository root**:

```bash
cd EvidentialSemanticEntropy
python semantic_uncertainty/generate_answers.py [OPTIONS]
```

### Basic Experiment Command

```bash
python semantic_uncertainty/generate_answers.py \
    --num_samples=500 \
    --model_name=$MODEL \
    --dataset=$DATASET \
    --num_generations=5 \
    --random_seed=42 \
    --compute_kle
```


test experiment:
```bash
python semantic_uncertainty/generate_answers.py \
    --model_name=TinyLlama-1.1B-Chat-v1.0 \
    --dataset=trivia_qa \
    --num_samples=5 \
    --num_generations=3 \
    --random_seed=42 \
    --force_cpu

```

- `$MODEL` is one of:
  `Llama-2-7b`, `Llama-2-13b`, `Llama-2-7b-chat`, `Llama-2-13b-chat`,
  `falcon-7b`, `falcon-40b`, `falcon-40b-instruct`, `falcon-7b-instruct`,
  `Mistral-7B-v0.1`, `Mistral-7B-Instruct-v0.1`

- `$DATASET` is one of: `trivia_qa`, `squad`, `nq`, `svamp`

### With Evidential Semantic Entropy Ablations

```bash
python semantic_uncertainty/generate_answers.py \
    --num_samples=500 \
    --model_name=Llama-2-7b-chat \
    --dataset=trivia_qa \
    --num_generations=5 \
    --random_seed=42 \
    --compute_kle \
    --compute_ablations
```

---

## Key CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--model_name` | `Llama-2-7b-chat` | LLM to use for generation |
| `--dataset` | `trivia_qa` | Evaluation dataset |
| `--num_samples` | `400` | Number of dataset samples |
| `--num_generations` | `10` | High-temperature generations per sample |
| `--temperature` | `1.0` | Sampling temperature |
| `--num_few_shot` | `5` | Number of few-shot examples |
| `--random_seed` | `10` | Random seed |
| `--compute_kle` | `False` | Enable Kernel Language Entropy computation |
| `--compute_ablations` | `False` | Enable EvSemE ablation studies |
| `--use_context` | `False` | Include context in prompt (auto-True for svamp) |
| `--answerable_only` | `False` | Skip unanswerable questions (auto-True for squad) |
| `--enable_brief` | `True` | Use brief answer prompt |
| `--compute_p_true` | `True` | Compute p_true baseline |
| `--skip_generation` | `False` | Skip generation; only compute uncertainties |
| `--compute_uncertainties` | `True` | Compute uncertainty measures after generation |
| `--analyze_run` | `True` | Run analysis after uncertainty computation |

---

## Pipeline Workflow

The pipeline runs in two sequential phases:

### Phase 1: Answer Generation

- Loads dataset from HuggingFace
- Initializes the specified LLM
- Generates answers at low temperature (0.1) for accuracy
- Generates answers at high temperature for entropy estimation
- Saves: `train_generations.pkl`, `validation_generations.pkl`
- Logs to wandb

### Phase 2: Uncertainty Computation

- Computes semantic entropy, KLE, evidential semantic entropy, p_ik, p_true
- Optionally computes EvSemE ablations (`--compute_ablations`)
- Saves: `uncertainty_measures.pkl`
- Runs analysis and generates visualization outputs

To **skip generation and only recompute uncertainties** on an existing wandb run:

```bash
python semantic_uncertainty/generate_answers.py \
    --skip_generation \
    --eval_wandb_runid=YOUR_WANDB_RUN_ID
```

---

## Outputs

Outputs are written to `../EXP/` (relative to the working directory):

```
../EXP/<run_name>/
├── experiment_details.pkl
├── train_generations.pkl
├── validation_generations.pkl
├── uncertainty_measures.pkl
└── analysis/
```

All runs are also logged to Weights & Biases.

---

## Multi-GPU Configuration

Edit `configs/data_config.yaml` to set device mapping:

```yaml
device: "cpu"          # fallback device
hf_cache_dir: "hf_cache"
device_map:            # GPU layer allocation
  ...
```

---

## Running Only Uncertainty Computation

If you already have generation outputs (from a previous wandb run):

```bash
python semantic_uncertainty/compute_uncertainty_measures.py \
    --eval_wandb_runid=YOUR_WANDB_RUN_ID
```

---

## Analyzing Results

```bash
python semantic_uncertainty/analyze_results.py \
    --eval_wandb_runid=YOUR_WANDB_RUN_ID
```
