# SD-CDM: Strategy-Aware Dual-Channel Cognitive Diagnosis Model

> **Paper:** *Strategy-Aware Dual-Channel Cognitive Diagnosis Model (SD-CDM)*  
> **Status:** Under Review — *Expert Systems with Applications*

---

## Overview

SD-CDM addresses three structural limitations of existing deep learning-based Cognitive Diagnosis Models (CDMs):

1. **Conflation of noise and active misconceptions** — standard models map low scores to a single near-zero scalar, unable to distinguish "not yet learned" from "learned incorrectly."
2. **Strategy homogeneity assumption** — existing models assign static concept weights, ignoring individual problem-solving habits.
3. **Semantic gap for LLM integration** — opaque scalar outputs provide insufficient grounding for downstream language models.

SD-CDM solves these with two core modules:

| Module | Role |
|--------|------|
| **BiKD** (Bidirectional Knowledge Diagnostic) | Decouples student state into orthogonal *Positive Mastery* h⁺ and *Negative Misconception* h⁻ vectors, with a learnable per-exercise Trap Intensity coefficient μⱼ |
| **CAM** (Cognitive Attention Machine) | Driven by a latent *Student Habit Vector*, dynamically allocates attention weights over required concepts to simulate personalised cognitive activation |

---

## Architecture

```
Input (student_id, exercise_id, Q-matrix row)
        │
        ▼
┌─────────────────────────────────┐
│  Stage 1 · BiKD                 │
│  h⁺ = σ(E_pos[i])   h⁻ = σ(E_neg[i])   μⱼ = σ(mⱼ)  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Stage 2 · CAM                  │
│  α_ijk = softmax(MLP([v_habit ║ eⱼ ║ cₖ]) | Q-mask) │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Stage 3 · Prediction           │
│  Ŝᵢⱼ = Σₖ αᵢⱼₖ(h⁺ᵢₖ − μⱼh⁻ᵢₖ) − dⱼ   ŷ = σ(Ŝᵢⱼ) │
└─────────────────────────────────┘
```

---

## Quick Start

### Installation

```bash
pip install torch scikit-learn tqdm EduCDM
```

### Basic Usage

```python
from SDCDM import SDCDM

# Initialise model
model = SDCDM(
    knowledge_n=110,   # number of knowledge concepts
    exer_n=26660,      # number of exercises
    student_n=4151,    # number of students
    hidden_dim=64      # embedding dimension for CAM
)

# Train
model.train(
    train_data=train_loader,
    test_data=test_loader,
    epoch=20,
    device="cuda",
    lr=0.001,
    lam=0.1,      # weight of margin regularisation λ
    margin=0.1    # margin hyper-parameter ε
)

# Evaluate
auc, acc = model.eval(test_loader, device="cuda")
print(f"AUC: {auc:.4f}  ACC: {acc:.4f}")

# Save / load
model.save("sdcdm_assist09.pt")
model.load("sdcdm_assist09.pt")
```

### DataLoader Format

Each batch should yield a 4-tuple:

```python
(user_id, item_id, knowledge_emb, y)
# user_id       : LongTensor  (B,)
# item_id       : LongTensor  (B,)
# knowledge_emb : FloatTensor (B, K)  — Q-matrix row for the exercise
# y             : FloatTensor (B,)    — binary response label {0, 1}
```

This is identical to the format used by [EduCDM](https://github.com/bigdata-ustc/EduCDM).

### Extracting Diagnostic Profiles

```python
import torch

student_ids = torch.arange(100)   # first 100 students
profiles = model.get_student_profiles(student_ids, device="cuda")

# profiles["pos_mastery"]        → (100, K)  Positive Mastery h⁺
# profiles["neg_misconception"]  → (100, K)  Negative Misconception h⁻
# profiles["habit_vector"]       → (100, D)  Latent Habit Vector v_habit
```

These three tensors can be serialised as structured semantic priors for downstream LLMs.

---

## Datasets

Experiments are conducted on three public ASSISTments benchmarks:

| Dataset | Students | Exercises | Concepts | Interactions | Sparsity |
|---------|----------|-----------|----------|--------------|----------|
| ASSIST09 | 4,151 | 26,660 | 110 | 325,637 | 99.70% |
| ASSIST15 | 19,840 | 100 | 100 | 683,203 | 65.56% |
| ASSIST17 | 1,709 | 3,162 | 102 | 942,816 | 82.55% |

Download from the [ASSISTments website](https://sites.google.com/site/assistmentsdata/).

---

## Results (5-fold CV)

| Model | ASSIST09 AUC | ASSIST15 AUC | ASSIST17 AUC |
|-------|-------------|-------------|-------------|
| DINA | 0.7412 | 0.6585 | 0.6994 |
| IRT | 0.7511 | 0.6645 | 0.7890 |
| NCDM | 0.7572 | 0.6692 | 0.7641 |
| SCD | 0.7813 | 0.7021 | 0.7925 |
| **SD-CDM (Ours)** | **0.7972** | **0.7096** | **0.8000** |

---

## Hyper-parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 64 | Embedding dimension D for CAM |
| `lr` | 0.001 | Adam learning rate |
| `lam` | 0.1 | Margin regularisation weight λ |
| `margin` | 0.1 | Margin ε in L_reg |
| `weight_decay` | 1e-4 | L2 regularisation λ_Θ |
| Batch size | 256 | — |
| Early stopping | 10 epochs | Based on validation AUC |

Trap intensity is initialised from U(−2.0, −1.0), giving μⱼ ∈ (0.12, 0.27) at the start of training.

---

## Baseline

The NCDM baseline (`NCDM.py`) is included for direct comparison. It is taken from [Wang et al. (2022)](https://ieeexplore.ieee.org/document/9865139) via the EduCDM library.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{sdcdm2026,
  title   = {Strategy-Aware Dual-Channel Cognitive Diagnosis Model},
  author  = {[Authors]},
  journal = {Expert Systems with Applications},
  year    = {2026},
  note    = {Under Review}
}
```

---

## License

MIT
